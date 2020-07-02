# adapted from https://github.com/labstreaminglayer/liblsl-Python/blob/master/pylsl/examples/ReceiveData.py
"""Example program to show how to read a multi-channel time series from LSL."""

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylsl import StreamInlet, resolve_stream, local_clock
from communication2game import UdpCommunicator
import threading
import time
from pathlib import Path
import scipy.io as sio
from threading import Thread

#################################################################################################################
# UPDATE: The analysis causes delay for receiveEEG. Therefore we seperate the analysis and the receiveEEG.
# Now the analysis part is in an independent thread.
#################################################################################################################

# load channel names and ids from raw data file
data_folder = Path("datasets/")
eval_raw_fname = data_folder / 'BCICIV_eval_ds1a.mat'
eval_raw_chans = sio.loadmat(eval_raw_fname, squeeze_me=True)["nfo"]["clab"].item()


# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

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
    sample, timestamp = inlet.pull_chunk(max_samples=100) #pull_chunk separately
    recv_stamp = local_clock()
    samples_buffer += sample
    timestamps_buffer += timestamp
    samples_buffer = samples_buffer[-150:]
    timestamps_buffer = timestamps_buffer[-150:]
    timestamps_buffer = [x - recv_stamp for x in timestamps_buffer]
    return samples_buffer, timestamps_buffer


def animate(samples_buffer, timestamps_buffer, axs):
    """
    plots the current EEG data samples in "samples" with the timestamps in "timestamps"
    """
    # pull a chunk of data from the stream (we are pulling chunks to avoid having to sync the period of this animate
    # function with the actual sampling frequency)
    print("animating")
    if len(timestamps_buffer) > 0:
        if max(timestamps_buffer) < -2:  # if the delay is larger than 2 seconds, kill the app
            print("exiting because of too high delays...")
            exit()

        # Plot the received data
        axs.clear()
        axs.plot(timestamps_buffer, np.array(samples_buffer))
        axs.set(title='EEG signal from the LSL', xlabel='Time',
                    ylabel='Electrode potential(uV)')
        axs.legend(['C3', 'C4'])
        axs.tick_params(axis='x', rotation=45)

        plt.pause(0.001)

def online_analysis():
    global samples_buffer
    global timestamps_buffer
    while True:
        print("analysis")

        ### Instead of doing analysis in the while loop, we conduct it in a thread to avoid delay.

        # filtering
        # artifact removal
        # feature extraction
        # load classifier

        # with open('trained_linear_clf.pkl', 'rb') as f:
        #    clf = pickle.load(f)

        # predict class
        # transfer function

        #out comment the delay while doing analysis. It's simulating the analysis time in the test.
        # send control command to spelling interface
        # control_command: 1 = left, 2 = headlight, 3 = right
        # udps.send_command_raw(random.randint(0, 1), 1)
        time.sleep(2)

udps = UdpCommunicator.Sender()
udpr = UdpCommunicator.Receiver()
receiverthread = threading.Thread(target=udpr.receive_game_input, args=())
receiverthread.start()


analysisthread = Thread(target=online_analysis, args = ())
analysisthread.start()

while(True):
    samples_buffer, timestamps_buffer = receiveEEGsamples(inlet, samples_buffer_init, timestamps_buffer_init)
    animate(samples_buffer, timestamps_buffer, axs)
    # timestamps_buffer[-1]:: delay of the last sample
    # if len(timestamps_buffer) != 0:
    #     print("thread", timestamps_buffer[-1], len(timestamps_buffer))

    time.sleep(0.01)
