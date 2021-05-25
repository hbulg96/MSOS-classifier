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


def note_onset_peak_classifier(input_path, show_graph=False):

    input_file = read(input_path)  # read wav file
    fs = input_file[0]
    input_file = numpy.array(input_file[1], dtype = float)  # interpret file as numpy array
    print("Fs = ", fs)

    onset_envelope = librosa.onset.onset_detect(y=input_file, sr=fs, units='time')

    o_env = librosa.onset.onset_strength(y=input_file, sr=fs)
    times = librosa.times_like(o_env, sr=fs)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=fs)

    peaks, peak_properties = scipy.signal.find_peaks(o_env, prominence=(0.2, 10))


    # PEAK DISTANCES
    distances_between_all_peaks = []
    # find mean distance between peaks

    if len(peaks) >= 2:
        for x in range(len(peaks)-1):
            initial_peak = peaks[x]  # find first peak
            second_peak = peaks[x+1]  # find end peak

            distance_between_peaks = second_peak - initial_peak  # distance between peaks (in time (frequency bins))
            distances_between_all_peaks.append(distance_between_peaks)
            pass

        mean_distance_between_peaks = sum(distances_between_all_peaks)/len(distances_between_all_peaks)  # mean distance between peaks
        sd_distance_between_peaks = numpy.std(distances_between_all_peaks)  # standard deviation of distance between peaks

        pass

    elif len(peaks) < 2:

        mean_distance_between_peaks = 0
        sd_distance_between_peaks = 0
        pass

    # PEAK PROMINENCE
    prominences = scipy.signal.peak_prominences(o_env, peaks)[0]
    SD_peak_prominence = sum(prominences)/len(prominences)
    # PEAK HEIGHT (pretty much same as peak prominence!)
    peak_heights = o_env[peaks]
    mean_peak_height = sum(peak_heights)/len(peak_heights)


    # PEAK CREST FACTORS
    peak_crest_factors = []
    peak_window = 10  # 50 samples long VARIABLE
    buffer_length = int(round((peak_window/2), 0))
    for v in range(buffer_length):
        o_env = numpy.insert(o_env, 0, 0) # at first location, add a 0 value
        o_env = numpy.append(o_env, 0)  # at end location, add a 0 value
        pass

    for x in range(len(peaks)):
        
        peak_val = peaks[x]  # onset value at peak location
        # find window of note onset values around the peak location
        window_of_onset_values  = []
        
        for y in range(peak_window):

            peak_val_location_in_window = buffer_length  # location of peak value in the (usually 50 length) window, half window length
            current_location_in_window = peak_val_location_in_window - y  # location of current value in window

            current_location_in_onset = current_location_in_window + peak_val  # location of current value in larger note onset array, o_env
            current_val_in_onset = o_env[current_location_in_onset]  # value of note onset in o_env array

            window_of_onset_values.append(current_val_in_onset)  # append value to "window" array
            pass

        #calculate window rms
        squared_window_values = []
        for z in range(len(window_of_onset_values)):
            squared_onset_value_in_window = window_of_onset_values[z]  # find square of note onset value in window
            squared_window_values.append(squared_onset_value_in_window)
            pass

        mean_of_squares_of_window_values = sum(squared_window_values)/len(squared_window_values)  # find mean of squared window values
        window_rms_val = math.sqrt(mean_of_squares_of_window_values)  # find RMS value of window around note onset peak
        
        peak_crest_factor = peak_val/window_rms_val
        peak_crest_factors.append(peak_crest_factor)
        pass

    #print("Peaks = ", peaks)
    #print("Crest factors of peaks = ", peak_crest_factors)

    sd_of_peak_crest_factors = numpy.std(peak_crest_factors)
    mean_of_peak_crest_factors = sum(peak_crest_factors)/len(peak_crest_factors)

    
    
    feature_1 = sd_of_peak_crest_factors
    feature_2 = mean_of_peak_crest_factors

    print("SD of peak crest factors = ", feature_1)
    print("Mean of peak crest factors = ", feature_2)
    
    if show_graph == True:
        D = numpy.abs(librosa.stft(input_file))
        fig, ax = pyplot.subplots(nrows=2, sharex=True)
        pyplot.figure(1)
        librosa.display.specshow(librosa.amplitude_to_db(D, ref=numpy.max), sr=fs,
                                 x_axis='time', y_axis='log', ax=ax[0])
        ax[0].set(title='Power Spectrogram')
        ax[0].label_outer()
        ax[1].plot(times, o_env, label='Onset Strength')
        ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
                     linestyle='--', label='Onsets')
        ax[1].legend()

        pyplot.figure(2)
        pyplot.plot(o_env)
        pyplot.plot(peaks, o_env[peaks],'x')

        print("Peaks = ", peaks)
        print("o_env[peaks] = ", o_env[peaks])
        pyplot.show()
        pass

    elif show_graph == False:
        pass
    else:
        print("Error in detecting show_graph variable")
        pass

    return(feature_1, feature_2)




"""
test_file = read(r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\Effects\0M8.wav")
test_file = numpy.array(test_file[1], dtype = int)
matplotlib.pyplot.plot(test_file)
pyplot.xlabel("Time")
pyplot.ylabel("Amplitude")
pyplot.show()
"""

"""
matplotlib.pyplot.plot(gain_boosted_file)
pyplot.xlabel("Time")
pyplot.ylabel("Amplitude")
pyplot.show()
"""
"""
f, t, Sxx = signal.spectrogram(average_effect_file, 44100)
pyplot.pcolormesh(t, f, Sxx, shading='gouraud')
pyplot.ylabel('Frequency [Hz]')
pyplot.xlabel('Time [sec]')
pyplot.show()
"""
        

