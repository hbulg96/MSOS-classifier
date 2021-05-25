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

def likelihood_of_class_based_on_rhythm(category, in_packet_weight, between_packet_weight, packet_rhythm_tendency, beat_rhythm_tendency):
    bias_factor = 0  #  + 1 bias factor for negative weights
    #print("Category = ", category)
    
    if in_packet_weight < 0:
        bias_factor = 1
        pass

    else:
        pass
    weight1 = (in_packet_weight * packet_rhythm_tendency) + bias_factor
    
    bias_factor = 0
    if between_packet_weight < 0:
        bias_factor = 1
        pass

    else:
        pass
    
    weight2 = (between_packet_weight * beat_rhythm_tendency) + bias_factor

    likelihood_of_class = (weight1/2) + (weight2 / 2)
    
    #print("Likelihood of class = ", likelihood_of_class)
    return(likelihood_of_class)


def Classification_layer_1(input_path,
                           input_category,
                           peak_detection_window_length=4000,
                           input_crest_factor=4.1,
                           nearest_value_cutoff=400,
                           peak_flatten_style="max",
                           flat_peak_similarity_magnitude=0.7,
                           rhythmic_packet_nearest_value_cutoff=6000,
                           plot_results=False):
    
    input_file = read(input_path)

    # DETECT PEAKS
    # This function detects peak points in an input file based on user input window length and crest factor

    initial_peak_points, initial_peak_values = peakdetection.peakdetection(input_file,
                                                                           window_length = peak_detection_window_length,
                                                                           crest_factor=input_crest_factor)

    if len(initial_peak_points) == 0:
        
        print("No peaks detected")

        packet_rhythm_tendency = 0
        beat_rhythm_tendency = 0
        pass

    else:

        peak_file = numpy.array((initial_peak_points, initial_peak_values), dtype = int) # create numpy array of peak points


        # FLATTEN PEAKS
        # This function flattens nearby peak points into "single" peaks
        # This will make it easier to detect rhythmic points and the starting point of impulses

        flattened_peak_points, flattened_peak_values = peakflatten.peakflatten(peak_file,
                                                                               nearest_value_cutoff= nearest_value_cutoff,
                                                                               peak_centre=peak_flatten_style,
                                                                               all_positive=True)


        #DETECT RHYTHM
        # This function collects points of similar amplitude together
        similar_magnitude_locations,similar_magnitude_values = find_similar_magnitude_peaks.find_similar_magnitude_peaks(
                                                                                                                        flattened_peak_points,
                                                                                                                        flattened_peak_values,
                                                                                                                        magnitude_similarity=flat_peak_similarity_magnitude)
            
        # This function collects point of similar magnitude into rhythmic "packets"
        # It uses only one layer of packet detection but in theory could endlessly iterate
        # through nearby points with smaller cutoff values finding subdivisions of "rhythmic" peaks
        rhythmic_packet_locations, rhythmic_packet_values = find_rhythmic_packets.find_rhythmic_packets(
                                                                                similar_magnitude_locations,
                                                                                  similar_magnitude_values,
                                                                                  nearest_peak_cutoff=rhythmic_packet_nearest_value_cutoff)

        packet_rhythm_tendency, beat_rhythm_tendency = rhythmdetection.rhythmdetection(rhythmic_packet_locations,
                                                                                        rhythmic_packet_values,
                                                                                       )


        # weight in-packet rhythm tendency and between-packet rhythm tendency for music, nature and urban
        # weight number represents how important each of the rhythm tendencies are in that class
        
        if plot_results == True:
            input_file = numpy.array(input_file[1], dtype = int)  # create numpy array of "original" input file
            matplotlib.pyplot.plot(input_file, label= "input_file")  # plot input waveform
            matplotlib.pyplot.plot(initial_peak_points,initial_peak_values, label= "peak_values")  # plot detected peak values
            pyplot.xlabel("Time")
            pyplot.ylabel("Amplitude")
            #pyplot.legend()

            for similar_peak_loc, similar_peak_val in zip(similar_magnitude_locations, similar_magnitude_values):
                matplotlib.pyplot.plot(similar_peak_loc, similar_peak_val, label= "similar_peak_values", marker="o")
                pass

                
            #matplotlib.pyplot.plot(flattened_peak_points, flattened_peak_values, label= "flat_peak_values", marker="x")  # plot flattened peak values
            pyplot.xlabel("Time")
            pyplot.ylabel("Amplitude")
            #pyplot.legend()
            pyplot.show()
            pass

        else:
            pass

        pass

    return(packet_rhythm_tendency, beat_rhythm_tendency)


    
def Predict_Category(packet_rhythm_tendency, beat_rhythm_tendency, in_packet_weights, between_packet_weights, print_values=True):

    sound_categories = ["Music", "Nature", "Urban", "Human", "Effects"]

    all_predictions = []

    if print_values == True:
        print("Packet rhythm tendency = ", packet_rhythm_tendency)
        print("Beat rhythm tendency = ", beat_rhythm_tendency)
        pass
    else:
        pass
    
    for x in range(len(sound_categories)):

        sound_category = sound_categories[x]
        in_packet_weight = in_packet_weights[x]
        between_packet_weight = between_packet_weights[x]
        
        prediction_of_class = likelihood_of_class_based_on_rhythm(sound_category,
                                            in_packet_weight,
                                            between_packet_weight,
                                            packet_rhythm_tendency,
                                            beat_rhythm_tendency)

        if print_values == True:
            print(sound_category, " prediction = ", prediction_of_class)
            pass
        else:
            pass

        all_predictions.append(prediction_of_class)
        pass

    most_likely_category_value = max(all_predictions)

    category_prediction = sound_categories[(all_predictions.index(most_likely_category_value))]

    return(category_prediction)
