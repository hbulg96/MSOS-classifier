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
    bias_factor = 0
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

    likelihood_of_class = weight1+weight2
    
    #print("Likelihood of class = ", likelihood_of_class)
    return(likelihood_of_class)


def generate_random_input_file(top_of_directory_path):

    path = top_of_directory_path
    directory = os.listdir(path)

    sound_categories = ["Music", "Nature", "Urban", "Human", "Effects"]

    category_index = random.randint(0, 4)
    sound_category = sound_categories[category_index]

    #print("Input file category = ", sound_category)

    path = top_of_directory_path + sound_category + "\\"
    
    wav_file_index = random.randint(0, 299)  # generate random input file from category
    directory = os.listdir(path)  # make a directory list
    directory.sort()  # sort list
    input_file = directory[wav_file_index]  # index random number into directory
    #print("Input filename = ", input_file)
    current_filename = str(path) + "\\" + input_file
    
    input_path = current_filename
    category = sound_category
    
    #print("Input file category = ", sound_category)
    return(input_path, category)



def Classification_layer_1(input_path, input_category):
    
    input_file = read(input_path)
    #print("input_file= ", input_file)
    #print(len(input_file))



    #peak detection variables
    peak_detection_window_length=4000
    input_crest_factor=4.1

    #peak "flattening" (collecting nearby peaks into single peaks) variables
    nearest_value_cutoff=400
    peak_flatten_style="max"

    #variables for collecting peaks of similar amplitude into lists
    flat_peak_similarity_magnitude = 0.7

    #variables for collecting similar amplitude peaks into rhythmic "packets" nearby to each other
    rhythmic_packet_nearest_value_cutoff = 6000







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
                                                                                        rhythmic_packet_values)


        # weight in-packet rhythm tendency and between-packet rhythm tendency for music, nature and urban
        # weight number represents how important each of the rhythm tendencies are in that class

        pass

    return(packet_rhythm_tendency, beat_rhythm_tendency)
    
def Predict_Category(packet_rhythm_tendency, beat_rhythm_tendency):

    sound_categories = ["Music", "Nature", "Urban", "Human", "Effects"]

    in_packet_weights =      [0.7, 0.5, 0.6, -0.2, -0.4]
    between_packet_weights = [0.8, 0.2, 0.6, -0.2, -0.4]

    all_predictions = []
    
    for x in range(len(sound_categories)):

        sound_category = sound_categories[x]
        in_packet_weight = in_packet_weights[x]
        between_packet_weight = between_packet_weights[x]
        
        prediction_of_class = likelihood_of_class_based_on_rhythm(sound_category,
                                            in_packet_weight,
                                            between_packet_weight,
                                            packet_rhythm_tendency,
                                            beat_rhythm_tendency)
        
        all_predictions.append(prediction_of_class)
        pass


    most_likely_category_value = max(all_predictions)

    category_prediction = sound_categories[(all_predictions.index(most_likely_category_value))]

    return(category_prediction)

def check_if_correct_category(category_prediction, file_category):

    #print("")
    #print("Category prediction = ", category_prediction)
    #print("Actual category = ", file_category)

    if category_prediction == file_category:

        prediction_correct = 1

        pass

    elif category_prediction != file_category:

        prediction_correct = 0

        pass

    else:
        print("error in detecting correct prediction")

        pass

    #print("Prediction correct = ", prediction_correct)


    """
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
    """

    exact_prediction = prediction_correct

    return(exact_prediction)

prediction_correct = 0

path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\\"

n = 0
print("Rhythm detection method")
for x in range(1000):
    input_path, file_category = generate_random_input_file(path)

    packet_rhythm_tendency, beat_rhythm_tendency = Classification_layer_1(input_path, file_category)
    category_prediction = Predict_Category(packet_rhythm_tendency, beat_rhythm_tendency)
    exact_prediction = check_if_correct_category(category_prediction, file_category)

    print("Prediction ", x, " of ", 100)
    prediction_correct += exact_prediction
    print("Number of predictions correct = ", prediction_correct, " of ", x)
    n+=1
    pass

print("Number of predictions correct = ", prediction_correct, " of ", n)

print("Random method")

prediction_correct = 0

n=0
for x in range(1000):
    input_path, file_category = generate_random_input_file(path)
    
    sound_categories = ["Music", "Nature", "Urban", "Human", "Effects"]
    category_index = random.randint(0, 4)
    sound_category = sound_categories[category_index]
    
    exact_prediction = check_if_correct_category(sound_category, file_category)
    print("Prediction ", x, " of ", 100)
    prediction_correct += exact_prediction
    print("Number of predictions correct = ", prediction_correct, " of ", x)
    n+=1
    pass

"""
filename = "45G.wav"
file_category = "Urban"
input_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\\"
input_path = input_path + file_category + "\\" + filename
output_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Python DSP files\Output data files\Moving average filter outputs\\"

Classification_layer_1(input_path, file_category)
"""

