# adapted from https://github.com/labstreaminglayer/liblsl-Python/blob/master/pylsl/examples/ReceiveData.py
"""Example program to show how to read a multi-channel time series from LSL."""

import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylsl import StreamInlet, resolve_stream, local_clock
# from communication2game import UdpCommunicator
import threading
import time
from pathlib import Path
import scipy.io as io
from threading import Thread
import mne
from mne.time_frequency import psd_multitaper
import socket
import pickle
import pyriemann
import sklearn



#################################################################################################################
# UPDATE: The analysis causes delay for receiveEEG. Therefore we seperate the analysis and the receiveEEG.
# Now the analysis part is in an independent thread.
#################################################################################################################

# load channel names and ids from raw data file
# data_folder = Path("datasets/")
# eval_raw_fname = data_folder / 'BCICIV_eval_ds1a.mat'
# eval_raw_chans = sio.loadmat(eval_raw_fname, squeeze_me=True)["nfo"]["clab"].item()
#HEADER = 64
#PORT = 5150
#PORT = 5151
#FORMAT = 'utf-8'
#DISCONNECT_MESSAGE = "quit"
#RECIEVER = "188.174.45.105" # Public IP address of Jin ask her to enable port forwarding for 5150
#ADDR = (RECIEVER, PORT)

#client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#client.connect(ADDR)

#clf = sklearn.pipeline

with open('C:/Users/Svea Marie Meyer/Desktop/BCI-Friendship-Lamps/trained_rieman.pkl', 'rb') as f:
    clf = pickle.load(f)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
print('found an eeg stream')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

# Turn on pyplot interactive mode in order to be able to update visualization continuously
plt.ion()

# Create figure for plotting
fig, axs = plt.subplots(1, 1)
samples_buffer = []
timestamps_buffer = []

samples_buffer_init = []
timestamps_buffer_init = []


def receiveEEGsamples(inlet, samples_buffer, timestamps_buffer):
    """
    Receives new EEG samples and timestamps from the LSL input and stores them in the buffer variables
    :param samples_buffer: list of samples(each itself a list of values)
    :param timestamps_buffer: list of time-stamps
    :return: updated samples_buffer and timestamps_buffer with the most recent 150 samples
    """
    sample, timestamp = inlet.pull_chunk(max_samples=100)  # pull_chunk separately
    recv_stamp = local_clock()
    samples_buffer += sample
    timestamps_buffer += timestamp
    samples_buffer = samples_buffer[-500:]
    timestamps_buffer = timestamps_buffer[-500:]
    timestamps_buffer = [x - recv_stamp for x in timestamps_buffer]
    return samples_buffer, timestamps_buffer


def animate(samples_buffer, timestamps_buffer, axs):
    """
    plots the current EEG data samples in "samples" with the timestamps in "timestamps"
    """
    # pull a chunk of data from the stream (we are pulling chunks to avoid having to sync the period of this animate
    # function with the actual sampling frequency)

    if len(timestamps_buffer) > 0:
        if max(timestamps_buffer) < -2:  # if the delay is larger than 2 seconds, kill the app
            print("exiting because of too high delays...")
            exit()

        # Plot the received data
        axs.clear()
        axs.plot(timestamps_buffer, np.array(samples_buffer))
        axs.set(title='EEG signal from the LSL', xlabel='Time',
                ylabel='Electrode potential(uV)')
        axs.tick_params(axis='x', rotation=45)

        plt.pause(0.001)


def plot_frequency(samples_buffer, timestamps_buffer, axs):
    if len(timestamps_buffer) > 0:
        if max(timestamps_buffer) < -2:  # if the delay is larger than 2 seconds, kill the app
            print("exiting because of too high delays...")
            exit()

            # plot the received data
            # axs[0].clear()
            # axs[0].plot(xs, np.array(ys))
            # axs[0].set(title='EEG signal from the LSL', xlabel='Time',
            #            ylabel='Electrode potential(uV)')
            # axs[0].legend(['C3', 'C4'])
            # axs[0].tick_params(axis='x', rotation=45)

            # create the mne RawArray object from data in order to apply filtering and power spectrum functions from mne:
            # https://mne.tools/stable/auto_examples/io/plot_objects_from_arrays.html#sphx-glr-auto-examples-io-plot-objects-from-arrays-py
            info = mne.create_info(
                ch_names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                          '18'], sfreq=250)
            raw = mne.io.RawArray(np.array(samples_buffer).T, info)
            raw.filter(7, 20, fir_design='firwin', picks=[4, 5, 6, 7])

            psds, freqs = psd_multitaper(raw, low_bias=True,
                                         fmin=7, fmax=20, proj=True, picks=[4, 5, 6, 7],
                                         n_jobs=1)
            psds = 10 * np.log10(psds)

            # plot the power spectrum
            axs.clear()
            axs.plot(freqs, psds.T)
            axs.set(title='Multitaper PSD', xlabel='Frequency',
                    ylabel='Power Spectral Density (dB)')
            axs.legend(
                ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'])
            axs.set_ylim(0, 100)
            axs.set_title('Power Spectrum')

def send_processed_result_of_unicorn(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(send_length)
    client.send(message)
    print(client.recv(2048).decode(FORMAT))


def online_analysis():
    global samples_buffer
    global timestamps_buffer
    while True:

        ### Instead of doing analysis in the while loop, we conduct it in a thread to avoid delay.

        # filtering
        # artifact removal
        # feature extraction
        # load classifier
        print(np.shape(samples_buffer))
        if len(samples_buffer) > 499:
            #np.asarray(samples_buffer[-500:-1].reshape(1, 18, 500))
            samples_buffer_np = np.asarray(samples_buffer)
            print(samples_buffer_np.shape)
            samples_buffer_np = samples_buffer_np[:, 1:9]
            print(samples_buffer_np.shape)
            samples_buffer_np = samples_buffer_np.reshape(1,8, 500)
            np.nan_to_num(samples_buffer_np, copy=False)
            print(samples_buffer_np.shape)
            classification_result = clf.predict(samples_buffer_np)


        # predict class
        # transfer function
            #classification_result = np.random.choice([0,1,2])
            states = ['purple for relaxed', 'blue for stressed', 'white for neutral']
            print(f'sending {states[int(classification_result)]}')
            #send_processed_result_of_unicorn(str(classification_result))
        # out comment the delay while doing analysis. It's simulating the analysis time in the test.
        # send control command to spelling interface
        # control_command: 1 = left, 2 = headlight, 3 = right
        # udps.send_command_raw(random.randint(0, 1), 1)

        else:
            print('signal too short')

        time.sleep(2)


# udps = UdpCommunicator.Sender()
# udpr = UdpCommunicator.Receiver()
# receiverthread = threading.Thread(target=udpr.receive_game_input, args=())
# receiverthread.start()


analysisthread = Thread(target=online_analysis, args=())
analysisthread.start()

while True:
    samples_buffer, timestamps_buffer = receiveEEGsamples(inlet, samples_buffer_init, timestamps_buffer_init)
    animate(samples_buffer, timestamps_buffer, axs)

    #alternatively plot power spectrum:
    plot_frequency(samples_buffer, timestamps_buffer, axs)

    # timestamps_buffer[-1]:: delay of the last sample
    # if len(timestamps_buffer) != 0:
    #     print("thread", timestamps_buffer[-1], len(timestamps_buffer))

    time.sleep(0.01)
