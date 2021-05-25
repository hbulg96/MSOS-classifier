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
from scipy import stats
from numpy import polyfit
import scipy.stats



def rms_skew_classifier(input_path, show_graph=False,
                        sd_low_lim=2000, skew_low_lim=2,
                        sd_hi_lim =3000, skew_hi_lim=4):

    input_file = read(input_path)  # read wav file
    fs = input_file[0]
    input_file = numpy.array(input_file[1], dtype = float)  # interpret file as numpy array

    number_of_desired_hop_frames = int(round((len(input_file)/2000), 0))
    file_rms_over_time = (librosa.feature.rms(input_file, frame_length=4096, hop_length=number_of_desired_hop_frames))[0]
    feature_1 = scipy.stats.skew(file_rms_over_time, nan_policy='omit')
    feature_2 = numpy.std(file_rms_over_time)

    if show_graph == True:
        if feature_2 > sd_low_lim and feature_1 > skew_low_lim:
            if feature_2 < sd_hi_lim and feature_1 < skew_hi_lim:
                print("RMS skewness = ", feature_1)
                print("RMS standard dev = ", feature_2)
                plt.figure(figsize=(14, 5))
                plt.plot(file_rms_over_time)
                plt.show()
                pass
            else:
                pass
            pass
        else:
            pass
        pass

    elif show_graph == False:
        print("RMS skew classifier = ", feature_1)
        print("RMS standard dev classifier = ", feature_2)
        pass

    else:
        print("No show_graph variable in rms_skew_classifier")
        pass
    pass

    rms_skew = feature_1
    rms_sd = feature_2

    return(rms_skew, rms_sd)

