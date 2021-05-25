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
from scipy import stats
from numpy import polyfit





def check_if_correct_category(category_prediction, file_category):

    print("Category prediction = ", category_prediction)
    print("Actual category = ", file_category)
    print("")

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
    exact_prediction = prediction_correct

    return(exact_prediction)




def generate_random_input_file(top_of_directory_path, input_index, category_index):

    exact_file_index = []
    
    path = top_of_directory_path
    directory = os.listdir(path)

    sound_categories = ["Music", "Nature", "Urban", "Human", "Effects"]
    #sound_categories = ["Music"]

    category_index = category_index
    exact_file_index.append(category_index)
    sound_category = sound_categories[category_index]

    print("Input file category = ", sound_category)

    path = top_of_directory_path + sound_category + "\\"
    
    wav_file_index = input_index  # generate random input file from category
    exact_file_index.append(wav_file_index)
    directory = os.listdir(path)  # make a directory list
    directory.sort()  # sort list
    input_file = directory[wav_file_index]  # index random number into directory
    #print("Input filename = ", input_file)
    current_filename = str(path) + "\\" + input_file
    
    input_path = current_filename
    category = sound_category
    
    #print("Input file category = ", sound_category)
    return(input_path, category, exact_file_index, input_file)




"""
Variables for rhythm detection algorithm
"""
# higher numbers in the weights mean that if there is a higher rhythmic "tendency" in the
# file, it will more likely be assigned to the respective category
sound_categories = ["Music", "Nature", "Urban", "Human", "Effects"]
# weighting assigned to rhythmic tendency within packets
in_packet_weights =      [0.7, 0.5, 0.6, 0.2, 0.1]
# weighting assigned to rhythmic tendency between grouped peaks (emulating beats)
between_packet_weights = [0.8, 0.2, 0.6, 0.2, 0.1]
#peak detection variables
peak_detection_window_length=12000
input_crest_factor=4
#peak "flattening" (collecting nearby peaks into single peaks) variables
nearest_value_cutoff=1500
peak_flatten_style="max"
#variables for collecting peaks of similar amplitude into lists
flat_peak_similarity_magnitude = 0.7
#variables for collecting similar amplitude peaks into rhythmic "packets" nearby to each other
rhythmic_packet_nearest_value_cutoff = 8000

plot_results= True

prediction_correct = 0

path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\\"


classification_method = "DSP"
rhythm_method = False
spectral_centroid_method = True
n = 0





def Single_File_Classification(input_path):

    #print("Single file selection method")
    #filename = "45G.wav"
    #file_category = "Urban"
    #input_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\\"
    #input_path = input_path + file_category + "\\" + filename
    #output_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Python DSP files\Output data files\Moving average filter outputs\\"

    
    if rhythm_method == True:
        # try to predict class based on rhythmic tendency of file
        packet_rhythm_tendency, beat_rhythm_tendency = classifier1.Classification_layer_1(input_path,
                                                                                          file_category,
                                                                                          peak_detection_window_length,
                                                                                          input_crest_factor,
                                                                                          nearest_value_cutoff,
                                                                                          peak_flatten_style,
                                                                                          flat_peak_similarity_magnitude,
                                                                                          rhythmic_packet_nearest_value_cutoff,
                                                                                          plot_results=plot_results)


        category_prediction = classifier1.Predict_Category(packet_rhythm_tendency,
                                                           beat_rhythm_tendency,
                                                           in_packet_weights,
                                                           between_packet_weights,
                                                           print_values=True)
        pass

    elif rhythm_method == False:
        pass

    else:
        print("error in detecting rhythm_method")
        pass


    if spectral_centroid_method == True:
        # try to predict class based on spectral centroid data
        # feed in input file
        # get out weights for each category
        average_sc, standard_deviation_sc = spectral_centroid_classifier.spectral_centroid_classifier(input_path,
                                                                                                        show_graph=False)

        #print("Average spectral centroid = ", average_sc)
        #print("Standard deviation of spectral centroid = ", standard_deviation_sc)
        pass

    elif spectral_centroid_method == False:
        pass

    else:
        print("error in detecting spectral_centroid_method")
        pass
    
    #exact_prediction = check_if_correct_category(category_prediction, file_category)

    return(average_sc, standard_deviation_sc)









if classification_method == "DSP":
    print("Rhythm detection method")

    previously_tested_file_indices = []

    all_heatmaps = []
    all_xedges = []
    all_yedges = []
    
    for n in range(5):
        #print("category index = ", n)
        average_spectral_centroids = []
        average_spectral_sds = []
        averages_over_sds = []

        category_heatmap = []
        category_xedges = []
        category_yedges = []
    
        for x in range(10):
            print("file index = ", x)
            input_path, file_category, exact_file_index, filename = generate_random_input_file(path, x, n)
            #print("File category = ", file_category)
            #print("Filename = ", filename)

            if exact_file_index in previously_tested_file_indices:
                print("Repeat file detected, index = ", exact_file_index)
                pass

            elif exact_file_index not in previously_tested_file_indices:

                if rhythm_method == True:
                    previously_tested_file_indices.append(exact_file_index)  # used to ensure we don't re-test file we've tested previously!

                    packet_rhythm_tendency, beat_rhythm_tendency = classifier1.Classification_layer_1(input_path,
                                                                                                      file_category,
                                                                                                      peak_detection_window_length,
                                                                                                      input_crest_factor,
                                                                                                      nearest_value_cutoff,
                                                                                                      peak_flatten_style,
                                                                                                      flat_peak_similarity_magnitude,
                                                                                                      rhythmic_packet_nearest_value_cutoff,
                                                                                                      plot_results=plot_results)


                    category_prediction = classifier1.Predict_Category(packet_rhythm_tendency,
                                                                       beat_rhythm_tendency,
                                                                       in_packet_weights,
                                                                       between_packet_weights,
                                                                       print_values=True)

                elif rhythm_method == False:
                    pass


                if spectral_centroid_method == True:
                    average_sc, sd_sc = Single_File_Classification(input_path)
                    average_spectral_centroids.append(average_sc)
                    average_spectral_sds.append(sd_sc)
                    a_over_sd = average_sc / sd_sc
                    averages_over_sds.append(a_over_sd)

                    pass

                elif spectral_centroid_method == False:
                    pass

                
                #exact_prediction = check_if_correct_category(category_prediction, file_category)

                #print("Prediction ", x, " of ", 1000)
                #print("File category = ", file_category)
                #prediction_correct += exact_prediction
                #print("Number of predictions correct = ", prediction_correct, " of ", x)
                pass

            else:
                print("Error in detecting exact file index")
                pass
            
            pass
        

        """
        # correlation between average and standard deviation spectral centroid values
        pearson_correlation_value = scipy.stats.pearsonr(average_spectral_centroids, average_spectral_sds)
         finds correlation in average and sd of spectral centroid
        print("")
        print("Pearson correlation value = ", pearson_correlation_value)
        pearson_correlations.append(pearson_correlation_value)
        """
        
        average_inverse_COV = sum(averages_over_sds)/len(averages_over_sds) # average of the inverse coefficients of variance!
        print("Average inverse Coeff of variance = ", average_inverse_COV)

        heatmap, xedges, yedges = numpy.histogram2d(average_spectral_centroids, average_spectral_sds, bins=20)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        category_heatmap.append(heatmap)
        category_xedges.append(xedges)        
        category_yedges.append(yedges)

        all_heatmaps.append(category_heatmap)
        all_xedges.append(category_xedges)
        all_yedges.append(category_yedges)
        
        matplotlib.pyplot.clf()
        matplotlib.pyplot.imshow(heatmap.T, extent=extent, origin='lower')
        
        print("Heatmap = ", heatmap)
        print("X edges = ", xedges)
        print("Y edges = ", yedges)
   
        
        pyplot.show()












elif classification_method == "Random":
    """
    Makes completely random guesses as a baseline for comparison
    """
    print("Random method")

    prediction_correct = 0

    n=0
    for x in range(1000):
        input_path, file_category,exact_file_index, filename = generate_random_input_file(path)
        
        sound_categories = ["Music", "Nature", "Urban", "Human", "Effects"]
        category_index = random.randint(0, 4)
        sound_category = sound_categories[category_index]
        exact_prediction = check_if_correct_category(sound_category, file_category)
        
        print("Prediction ", x, " of ", 100)
        prediction_correct += exact_prediction
        print("Number of predictions correct = ", prediction_correct, " of ", x)
        n+=1
        pass











elif classification_method == "Single":

    filename = "45G.wav"
    file_category = "Urban"
    input_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\\"
    input_path = input_path + file_category + "\\" + filename
    output_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Python DSP files\Output data files\Moving average filter outputs\\"
    Single_File_Classification(input_path)

        


















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
