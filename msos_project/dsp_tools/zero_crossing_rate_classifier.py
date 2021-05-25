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



def zero_crossing_rate_classifier(input_file, show_graph=False):

    input_file = read(input_file)  # read wav file
    fs = input_file[0]
    input_file = numpy.array(input_file[1], dtype = float)  # interpret file as numpy array

    zero_crossings_over_time = librosa.feature.zero_crossing_rate(input_file + 0.0001)
    zero_crossings_over_time = zero_crossings_over_time[0]

    average_zero_crossing_rate = sum(zero_crossings_over_time)/len(zero_crossings_over_time)
    standard_dev_zero_crossing_rate = numpy.std(zero_crossings_over_time)

    file_rms_over_time = librosa.feature.rms(input_file)  # find windowed rms in file
    file_rms = sum(file_rms_over_time[0])/len(file_rms_over_time[0])  # find energy average across file
    print("File rms = ", file_rms)

    if show_graph == True:
        print("Average zero crossing rate = ", average_zero_crossing_rate)
        print("Standard deviation of zero crossing rate = ", standard_dev_zero_crossing_rate)
        plt.figure(figsize=(14, 5))
        plt.plot(zero_crossings_over_time)
        plt.show()
        pass

    elif show_graph == False:
        pass

    else:
        print("No show_graph variable in zero_crossing_rate_classifier")
        pass
    pass


    return(average_zero_crossing_rate, standard_dev_zero_crossing_rate, file_rms)

