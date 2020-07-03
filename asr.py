import mne
import os
import numpy as np
import scipy.io as sio

import sys as sys
sys.path.append('..')
import asr
from scipy.stats import median_absolute_deviation
from numpy.linalg import pinv
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist, euclidean
from sklearn.utils.validation import check_array

#Artifact subspace reconstruction, based on code from
#https://github.com/bertrandlalo/timeflux_rasr/blob/master/timeflux_rasr/estimation.py
#https://github.com/sccn/clean_rawdata/blob/master/clean_asr.m
#https://github.com/nbara/python-meegkit/blob/master/meegkit/utils/asr.py


#ASR consists of three steps
#1: take a section of clean data
#2: convert to component space and find threshold
#3: remove components based on this threshold and reconstruct


class ASR():
    def __init__(self):
       
        self.sf = None
        self.picks = None
        self.n_channels = None
        self.x_c = None
        self.mixing = None
        self.threshold = None
        self.recon_prev = None
        self.counter = 0
        
    #1: take a section of clean data
    def clean_windows(self, raw, max_bad_chans = 0.2, z_param=[-3.5, 5.5], win_len = 1, win_overlap = 0.66, picks = None):
        
        self.sf = raw.info['sfreq']
        if picks == None:
            self.picks = mne.pick_types(raw.info, meg=False, eeg=True)
        else:
            self.picks = picks 
        self.n_channels = len(self.picks)

        n_samples = len(raw.times)
        win_samples = int(win_len*self.sf)
        offsets = np.int_(np.arange(0,  n_samples - win_samples, np.round(win_samples * (1 - win_overlap))))
        max_bad_chans = int(np.round(self.n_channels * max_bad_chans))
        
        #calculate RMS per channel over 1s windows
        rms_scores=[]
        for o in offsets:
            rms_scores.append(np.sqrt((raw[self.picks, o:o + win_samples][0] ** 2).mean(axis=1)))
 
        #next calculate z-score per window
        mu = np.mean(rms_scores,axis = 0)
        sig = np.std(rms_scores,axis = 0)
        z_scores = ((rms_scores - mu)/sig).T #shape = (n_channels, n_windows)
    
        # sort z scores into quantiles
        z_scores[np.isnan(z_scores)] = np.inf  # Nan to inf
        sz_scores = np.sort(z_scores, axis=0)
       
        #find windows with certain nr of channels exceeding thresholds
        mask_z = np.logical_or(sz_scores[-(np.int(max_bad_chans) + 1), :] > z_param[1], \
                               sz_scores[1 + np.int(max_bad_chans - 1), :] < z_param[0])
        #and any window with large fluctuations between channels
        mask_sd = np.logical_or(median_absolute_deviation(z_scores, axis=0) < .1, \
                                np.std(z_scores, axis=0) < .1)
        remove_mask = np.logical_or(mask_z, mask_sd)
        
        #remove affected windows
        sample_maskidx = []
        for win in np.where(remove_mask)[0]:
            sample_maskidx.append(np.arange(offsets[win], offsets[win] + win_samples))

        sample_mask2remove = np.unique(sample_maskidx)
        self.x_c= np.delete(raw[self.picks, :][0], sample_mask2remove, 1)
       
        return self.x_c
    
    #or if you have a clean data segment, use this
    def set_clean_windows(self, raw):
        
        self.x_c = raw[self.picks,:][0]
        print("set calibration data with "+ str(self.x_c.shape[0]) + " channels of len: " + str(self.x_c.shape[1]/100) + " s")
    
    #2: convert to component space and find threshold
    def calibrate(self, win_len = 0.5, win_overlap = 0.5, k=5):
        
        if self.x_c is None:
            print("First use clean_windows() or set_clean_windows() to set calibration data")
            return
        
        # Calculation of covariance matrix.
        cov_x = np.cov(self.x_c)
        l1_mean = geometric_median(cov_x.reshape((-1, self.n_channels * self.n_channels)))
        C = l1_mean.reshape((self.n_channels,self.n_channels))
        self.mixing = sqrtm(np.real(C))
        evals, evecs = np.linalg.eig(self.mixing)  # compute PCA
        indx = np.argsort(evals)  # sort in ascending
        evecs = evecs[:, indx]
        
        # Projection of the data into component space.
        y_c = np.dot(evecs.T, self.x_c)

        # Calculation of mean and std.dev of RMS values accross win_len second windows for each component i.
        n_samples = y_c.shape[1]
        win_samples= int(win_len * self.sf)
        offsets = np.int_(np.arange(0, n_samples - win_samples, np.round(win_samples * (1 - win_overlap))))

        rms_scores=[]
        for o in offsets:
            rms = np.sqrt(y_c[:,o:o+win_samples] ** 2).mean(axis=1)
            rms_scores.append(rms)
        
        #Determine threshold per component
        #Use median it's more robust
        sig= median_absolute_deviation(rms_scores,axis = 0)
        mu = np.median(rms_scores,axis = 0)
        self.threshold = mu + k * sig
        self.threshold = np.diag(self.threshold.dot(np.transpose(evecs)))

    #3: remove components based on this threshold and reconstruct
    def clean(self, raw, win_len=1.0,  win_overlap = 0.66, max_dims = 0.66):
        
        if self.threshold is None:
            print("First use calibrate() to calculate thresholds")
            return
        
        out_signal = np.zeros_like(raw.get_data(self.picks))
        n_samples = len(raw.times)
        win_samples = int(win_len*self.sf)
        offsets = np.int_(np.arange(0, n_samples - win_samples, np.round(win_samples * (1 - win_overlap))))
        recon_prev = None
        blend = (1 - np.cos(np.pi * np.arange(win_samples) / win_samples)) / 2
     
        for o in offsets:
       
            window = raw[self.picks, o:o+win_samples][0]

            #eigenvalue decomposition of covariance matrix
            cov_window = np.cov(window)

            #Check if we can use eigh instead of eig! makes use of symmetric properties and returns sorted eigenvalues
            evals, evecs = np.linalg.eig(cov_window)  # compute PCA
            indx = np.argsort(evals)  # sort in ascending
            evecs = np.real(evecs[:, indx])

            keep = (np.real(evals[indx]) < np.sum(self.threshold.dot(evecs) ** 2, axis = 0 )) 

            #don't discard first components
            max_art = int(max_dims * self.n_channels)
            keep += (np.arange(self.n_channels) < self.n_channels - max_art)
            keep = np.expand_dims(keep, 0)    

            if np.sum(keep)==self.n_channels:
                recon = np.eye(self.n_channels)
            else:
                v_trunk_mixing = pinv(keep.T * evecs.T.dot(self.mixing))
                recon = self.mixing.dot(v_trunk_mixing).dot(evecs.T)


            #Use raised cosine blending to smoothen transitions
            if recon_prev is not None:
                clean = blend * np.dot(recon, window) + (1 - blend) * np.dot(recon_prev, window)
            else:
                clean = np.dot(recon, window)

            out_signal[:,o:o+win_samples] = clean
            recon_prev = recon

        return out_signal
    
    def clean_epochs(self, epochs, win_len=0.5,  win_overlap = 0.66, max_dims = 0.66):
        
        if self.threshold is None:
            print("First use calibrate() to calculate thresholds")
            return
        
        
        raw = epochs.get_data(self.picks)[0]
        out_signal = np.zeros_like(raw)
        n_samples = raw.shape[1]
        win_samples = int(win_len*self.sf)
        offsets = np.int_(np.arange(0, n_samples - win_samples, np.round(win_samples * (1 - win_overlap))))
      
        blend = (1 - np.cos(np.pi * np.arange(win_samples) / win_samples)) / 2
     
        for o in offsets:
       
            window = raw[:, o:o+win_samples]

            #eigenvalue decomposition of covariance matrix
            cov_window = np.cov(window)

            #Check if we can use eigh instead of eig! makes use of symmetric properties and returns sorted eigenvalues
            evals, evecs = np.linalg.eig(cov_window)  # compute PCA
            indx = np.argsort(evals)  # sort in ascending
            evecs = np.real(evecs[:, indx])

            keep = (np.real(evals[indx]) < np.sum(self.threshold.dot(evecs) ** 2, axis = 0 )) 

            #don't discard first components
            max_art = int(max_dims * self.n_channels)
            keep += (np.arange(self.n_channels) < self.n_channels - max_art)
            keep = np.expand_dims(keep, 0)    

            if np.sum(keep)==self.n_channels:
                recon = np.eye(self.n_channels)
            else:
                v_trunk_mixing = pinv(keep.T * evecs.T.dot(self.mixing))
                recon = self.mixing.dot(v_trunk_mixing).dot(evecs.T)


            #Use raised cosine blending to smoothen transitions
            if self.recon_prev is not None:
                clean = blend * np.dot(recon, window) + (1 - blend) * np.dot(recon_prev, window)
            else:
                clean = np.dot(recon, window)

            out_signal[:,o:o+win_samples] = clean
            self.recon_prev = recon

        return out_signal

    
    
     
    def window_reconstruct(self, data, sf, win_len=1,  max_dims = 0.66, reconstruct_every = 32):
        
        if self.threshold is None:
            print("First use calibrate() to calculate thresholds")
            return

        window = data
        out_signal = np.zeros_like(window)
        n_samples = window.shape[1]
        win_samples = int(win_len*sf)

        blend = (1 - np.cos(np.pi * np.arange(win_samples) / win_samples)) / 2

        #eigenvalue decomposition of covariance matrix
        cov_window = np.cov(window)

        #Check if we can use eigh instead of eig! makes use of symmetric properties and returns sorted eigenvalues
        evals, evecs = np.linalg.eig(cov_window)  # compute PCA
        indx = np.argsort(evals)  # sort in ascending
        evecs = np.real(evecs[:, indx])

        keep = (np.real(evals[indx]) < np.sum(self.threshold.dot(evecs) ** 2, axis = 0 )) 

        #don't discard first components
        max_art = int(max_dims * self.n_channels)
        keep += (np.arange(self.n_channels) < self.n_channels - max_art)
        keep = np.expand_dims(keep, 0)    

        if np.sum(keep)==self.n_channels:
            recon = np.eye(self.n_channels)
        else:
            v_trunk_mixing = pinv(keep.T * evecs.T.dot(self.mixing))
            recon = self.mixing.dot(v_trunk_mixing).dot(evecs.T)


        #Use raised cosine blending to smoothen transitions
        if self.recon_prev is not None:
            clean = blend * np.dot(recon, window) + (1 - blend) * np.dot(recon_prev, window)
        else:
            clean = np.dot(recon, window)
        
        recon_prev = recon
        
        return clean
        

#calculate geometric median, used to get artifact robust covariance matrix
#From: https://github.com/bertrandlalo/timeflux_rasr/blob/master/utils/utils.py
def geometric_median(X, eps=1e-10, max_it=1000):
    """"
    Implementation of
    Vardi, Y., Zhang, C.H., 2000. The multivariate L1-median and associated data depth. Proc. Natl. Acad.
    Sci. U.S.A. 97, 1423–1426. https://doi.org/10.1073/pnas.97.4.1423
    founded here (tested)
    https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_features)
        n_features-dimensional points.
    eps : float (default: 1e-10)
        tolerance criterion
    max_it : int (default: 1000)
        maximum of iterations

    Returns
    -------
    X_median : ndarray, shape (n_features, )
        n_features-dimensional median of points X.

    """
    check_array(X)
    y = np.mean(X, 0)
    it = 0
    while it < max_it:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]  # unfortunately this algorithm doesn't handle 0-distance points

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1
        it += 1
    else:
        print("Geometric median could converge in %i iteration with eps=%.10f " % (it, eps))