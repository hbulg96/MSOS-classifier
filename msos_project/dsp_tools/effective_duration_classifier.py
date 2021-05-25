import numpy
import matplotlib
from matplotlib import pyplot
import scipy
from scipy import signal
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import os
import timeit
import traceback
import msos_project
from msos_project import *
from msos_project.dsp_tools import *
import msos_project.dsp_tools.peakdetection as peakdetection
import msos_project.dsp_tools.peakflatten as peakflatten
import msos_project.dsp_tools.rhythmdetection as rhythmdetection
import msos_project.dsp_tools.find_similar_magnitude_peaks as find_similar_magnitude_peaks
import msos_project.dsp_tools.find_rhythmic_packets as find_rhythmic_packets
from numpy import random
import msos_project.classification_1_rhythm_time_domain_v0_standalone as classifier1
import msos_project.dsp_tools.spectral_centroid_classifier as spectral_centroid_classifier
import msos_project.dsp_tools.spectral_centroid_assign_weights as spectral_centroid_assign_weights
import msos_project.dsp_tools.zero_crossing_rate_classifier as zero_crossing_rate_classifier
import msos_project.dsp_tools.rms_variation_classifier as rms_variation_classifier
from scipy import stats
from numpy import polyfit
import librosa
from librosa import *
from librosa import display
import scipy.stats
import math


def effective_duration_classifier(input_path, show_graph=False):

    input_file = read(input_path)  # read wav file
    fs = input_file[0]
    input_file = numpy.array(input_file[1], dtype = float)  # interpret file as numpy array
    print("Fs = ", fs)

    number_of_desired_hop_frames = int(round((len(input_file)/2000), 0))
    number_of_samples_in_file = len(input_file)
    number_of_time_bins = number_of_samples_in_file / number_of_desired_hop_frames
    #print("Number of desired hop frames =", number_of_desired_hop_frames)
    #print("Number of samples in file = ", number_of_samples_in_file)
    #print("Number of time bins = ", number_of_time_bins)
    
    file_rms_over_time = (librosa.feature.rms(input_file, frame_length=4096, hop_length=number_of_desired_hop_frames))[0]

    #print("rms = ", file_rms_over_time)

    max_rms_value = max(file_rms_over_time)
    max_rms_value_location = ((numpy.where(file_rms_over_time == max_rms_value))[0])[0]

    # find rms value that has 70% (variable) of max energy

    attack_energy_cutoff = 0.9

    cutoff_rms_value = max_rms_value*attack_energy_cutoff  # RMS energy that is percentage of max

    first_cutoff_value_index = ((numpy.where((file_rms_over_time > cutoff_rms_value) & (file_rms_over_time < max_rms_value)))[0])[0]
    first_cutoff_rms_value = file_rms_over_time[first_cutoff_value_index]

    # find start of attack RMS value

    start_energy_cutoff = 0.15

    start_rms_value = max_rms_value*start_energy_cutoff  # RMS energy that is percentage of max

    max_start_energy = 0.8
    max_start_energy_value = max_start_energy*max_rms_value

    first_start_value_index = ((numpy.where((file_rms_over_time > start_rms_value) & (file_rms_over_time < max_start_energy_value)))[0])[0]
    first_start_rms_value = file_rms_over_time[first_start_value_index]

    # find gradient between start and cutoff RMS values
    # difference in height/difference in width to find simple gradient

    y_difference = first_cutoff_rms_value - first_start_value_index
    x_difference = first_cutoff_value_index - first_start_value_index

    attack_gradient = y_difference/x_difference

    # log attack time (CUIDADO audio features)

    log_attack_time = math.log10(abs(first_cutoff_value_index - first_start_value_index))

    #print("rms = ", file_rms_over_time)

    # Effective duration (CUIDADO audio features)
    file_rms_over_time = numpy.insert(file_rms_over_time, 0, 0)

    effective_duration_cutoff = 0.5  # percentage of maximum to consider the "sustain" part of the signal
    effective_duration_cutoff_val = effective_duration_cutoff*max_rms_value  # find exact value by multiplying percentage by maximum RMS
    #first_duration_value_index = ((numpy.where((file_rms_over_time > effective_duration_cutoff_val) & (file_rms_over_time < max_rms_value)))[0])[0]
    #effective_duration_cutoff_val = file_rms_over_time[first_duration_value_index]
    
    effective_duration = 0
    y_axis_val = []
    
    for rms_val_index in range(len(file_rms_over_time)):
        rms_value = file_rms_over_time[rms_val_index]  # find RMS value at current index
        y_axis_val.append(effective_duration_cutoff_val)
        
        if rms_value > effective_duration_cutoff_val:
            # if RMS is higher than cutoff value, increase effective duration by 1 time unit
            effective_duration += 1
            pass

        elif rms_value <= effective_duration_cutoff_val:
            # if below cutoff val, do not increase effective duration value
            pass

        else:
            print("Error in detecting effective_duration_cutoff_val")
            pass
        pass     


    
    feature_1 = effective_duration
    feature_2 = log_attack_time
    
    if show_graph == True:
        print("First starting value index = ", first_start_value_index)
        print("First starting RMS value = ", first_start_rms_value)
        print("First cutoff value index = ", first_cutoff_value_index)
        print("First cutoff RMS value = ", first_cutoff_rms_value)
        print("Max RMS value = ", max_rms_value)
        print("Max RMS value location = ", max_rms_value_location)
        print("Effective duration = ", feature_1)
        print("Log attack time = ", feature_2)
        
        pyplot.plot(file_rms_over_time)
        pyplot.scatter(first_start_value_index, first_start_rms_value)
        pyplot.scatter(first_cutoff_value_index, first_cutoff_rms_value)
        pyplot.scatter(max_rms_value_location, max_rms_value)
        pyplot.plot(y_axis_val)
        pyplot.show()
        pyplot.show()
        pass

    elif show_graph == False:
        pass
    else:
        print("Error in detecting show_graph variable")
        pass

    return(feature_1, feature_2)


 

