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



def spectral_contrast_mean_classifier(input_path, show_graph=False):

    input_file = read(input_path)  # read wav file
    fs = input_file[0]
    input_file = numpy.array(input_file[1], dtype = float)  # interpret file as numpy array
    print("Fs = ", fs)

    feature_1 = librosa.feature.spectral_contrast(input_file, n_bands=8, fmin=100, sr=fs)
    feature_2 = librosa.feature.spectral_contrast(input_file, n_bands=8, fmin=100, sr=fs)
    
    number_of_bands = feature_1.shape[0]
    length_of_contrast_values = feature_1.shape[1]

    # find most tonal or most noisy band
    band_averages = [] #store average spectral contrast value per band
    
    for freq_band in range(number_of_bands):
        current_band = feature_1[freq_band]
        band_average = sum(current_band)/len(current_band)
        band_averages.append(band_average)
        for contrast_value in range(len(current_band)):
            current_value = current_band[contrast_value]
            pass
        pass

    max_contrast_band = max(band_averages)
    max_contrast_band_index = band_averages.index(max_contrast_band)

    min_contrast_band = min(band_averages)
    min_contrast_band_index = band_averages.index(min_contrast_band)


    #print("band_averages = ", band_averages)
    #print("largest average contrast band value = ", max_contrast_band)
    #print("max contrast band index = ", max_contrast_band_index)

    #print("smallest average contrast band value = ", min_contrast_band)
    #print("minx contrast band index = ", min_contrast_band_index)


    # most important band (feature band)
    feature_band_index = max_contrast_band_index

    feature_band = feature_1[feature_band_index] # contrast band with the highest average contrast value,
    # representing the most interesting/intentional sound?

    # "least" important band (noise band)
    noise_band_index = min_contrast_band_index

    noise_band = feature_1[noise_band_index]


    # amount of time spent with max contrast in feature band (should be closest to feature band)
    time_spent_at_feature_band = 0
    time_spent_at_noise_band = 0

    max_contrast_all_bands = [] # location of the max spectral contrast at any time
    
    for value_index in range(length_of_contrast_values):
        # find index of current spectral contrast value
        contrast_values_per_band = []
        for freq_band in range(number_of_bands):
            # find max value in all bands
            current_band = feature_1[freq_band]

            #print("freq band index = ", freq_band)
            #print("spectral contrast values = ", current_band)

            contrast_values_per_band.append(current_band[value_index])
            pass

        max_contrast_value_band = max(contrast_values_per_band)
        mcvb_index = contrast_values_per_band.index(max_contrast_value_band)

        max_contrast_all_bands.append(mcvb_index)

        min_contrast_value_band = min(contrast_values_per_band)
        mincvb_index = contrast_values_per_band.index(min_contrast_value_band)

        if mcvb_index == feature_band_index:
            time_spent_at_feature_band += 1
            pass
        else:
            pass

        if mincvb_index == noise_band_index:
            time_spent_at_noise_band += 1
            pass
        else:
            pass

    feature_1 = sum(feature_band)/len(feature_band) # feature band spectral contrast mean
    feature_2 = numpy.std(feature_band)/feature_1  # # coefficient of variation
        

    
    if show_graph == True:
        print("Noise-min metric = ", feature_1)
        print("Feature-max metric = ", feature_2)
        pyplot.figure(1)
        contrast_bands = librosa.feature.spectral_contrast(input_file, n_bands=8, fmin=100, sr=fs)
        pyplot.imshow(contrast_bands, aspect='auto', origin="lower", cmap="coolwarm")
        pyplot.ylabel('Frequency Band')
        pyplot.xlabel('Time (DFT bin)')
        pyplot.title("Spectral Contrast")

        # add lines for feature band and noise band
        contrast_bands = librosa.feature.spectral_contrast(input_file, n_bands=8, fmin=100, sr=fs)
        feature_band_x_points = [0, (contrast_bands.shape[1] - 1)]
        feature_band_y_points = [feature_band_index, feature_band_index]
        pyplot.plot(feature_band_x_points, feature_band_y_points, color='r',linewidth=3, label='feature band')

        noise_band_x_points = [0, (contrast_bands.shape[1] - 1)]
        noise_band_y_points = [noise_band_index, noise_band_index]
        pyplot.plot(noise_band_x_points, noise_band_y_points, color='b', linewidth=3, label='noise band')

        pyplot.plot(range(len(max_contrast_all_bands)), max_contrast_all_bands, color='g', label='max spectral contrast value')

        # plot feature and noise bands in their own graphs
        # feature band
        pyplot.figure(2)
        pyplot.plot(feature_band)
        pyplot.ylabel('Spectral Contrast')
        pyplot.xlabel('Time (DFT bins)')
        pyplot.title("Feature Band spectral contrast values")        

        pyplot.legend()
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
        

