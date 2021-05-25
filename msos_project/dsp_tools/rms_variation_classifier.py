import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import os
import timeit
import traceback
import msos_project
import librosa
from librosa import *
from librosa import display



def rms_variation_classifier(input_file, show_graph=False):

    input_file = read(input_file)  # read wav file
    fs = input_file[0]
    input_file = numpy.array(input_file[1], dtype = float)  # interpret file as numpy array

    number_of_desired_hop_frames = int(round((len(input_file)/8), 0))

    file_rms_over_time = (librosa.feature.rms(input_file, frame_length=4096, hop_length=number_of_desired_hop_frames))[0]  # find windowed rms in file

    window_values = []

    for n in range(len(file_rms_over_time)):
        window_values.append(n)
        pass

    if show_graph == True:
        print("File rms = ", file_rms_over_time)
        print("Window values = ", window_values)
        plt.scatter(window_values, file_rms_over_time)
        plt.show()
        pass

    elif show_graph == False:
        pass

    else:
        print("No show_graph variable in zero_crossing_rate_classifier")
        pass
    pass

    rms_variation = file_rms_over_time

    return(rms_variation, window_values)

