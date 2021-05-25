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




def Single_File_Classification(input_path,
                               input_sound_category,
                               rhythm_method,
                               spectral_centroid_method,
                               zero_crossing_rate_method):

    # load variables into namespace

    #print("Single file selection method")
    #filename = "45G.wav"
    #file_category = "Urban"
    #input_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\\"
    #input_path = input_path + file_category + "\\" + filename
    #output_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Python DSP files\Output data files\Moving average filter outputs\\"
    
    """
    Rhythm Detection Method
    """
    
    if rhythm_method == True:
        import msos_project.variables.rhythm_detection_variables  as rd_vars
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


    """
    Spectral Centroid - Average and Standard deviation method
    """
    

    if spectral_centroid_method == True:

        import msos_project.variables.heatmap_variables  as heatmap_vars

        all_yedges = heatmap_vars.all_yedges
        all_xedges = heatmap_vars.all_xedges
        all_heatmaps = heatmap_vars.all_heatmaps
        
        # try to predict class based on spectral centroid data
        # feed in input file
        # get out weights for each category
        average_sc, standard_deviation_sc = spectral_centroid_classifier.spectral_centroid_classifier(input_path,
                                                                                                        show_graph=False)


        print("Average spectral centroid = ", average_sc)
        print("Standard deviation of spectral centroid = ", standard_deviation_sc)
        
        sc_category_weights = spectral_centroid_assign_weights.spectral_centroid_assign_weights(average_sc,
                                                                                            standard_deviation_sc,
                                                                                            all_heatmaps,
                                                                                            all_xedges,
                                                                                            all_yedges)


        #print("Average spectral centroid = ", average_sc)
        #print("Standard deviation of spectral centroid = ", standard_deviation_sc)
        
        pass

    elif spectral_centroid_method == False:
        pass

    else:
        print("error in detecting spectral_centroid_method")
        pass


    """
    Zero Crossing Rate method
    """

    if zero_crossing_rate_method == True:
        average_zcr, sc_zcr = zero_crossing_rate_classifier.zero_crossing_rate_classifier(input_path, show_graph=True)
        
        zcr_category_weights = [0,0,0,0,0]
        pass

    elif zero_crossing_rate_method == False:
        pass

    else:
        print("Error in detecting zero_crossing_rate_method")
        pass
    
    """
    category_weights = sc_category_weights
    print("Spectral centroid category weights = ", sc_category_weights)
    """
    category_weights = zcr_category_weights
    # here, average other weights from the other DSP layers before prediction
    
    category_prediction = CheckCategoryFromWeights(category_weights)
    print("Category prediction = ", category_prediction)

    return(category_prediction)



def CheckCategoryFromWeights(input_category_weights):

    sound_categories = ["Music", "Nature", "Urban", "Human", "Effects"]
    category_index = input_category_weights.index(max(input_category_weights))
    category_prediction = sound_categories[category_index]

    return(category_prediction)






prediction_correct = 0

path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\\"


classification_method = "DSP"
rhythm_method = False
spectral_centroid_method = False
zero_crossing_rate_method = True
n_index = 0



if classification_method == "DSP":
    print("Rhythm detection method")

    previously_tested_file_indices = []
    
    for n in range(5):
        print("category index = ", n)
        average_zcrs = []
        sd_zcrs = []

        
        for x in range(300):
            print("file index = ", x)
            input_path, file_category, exact_file_index, filename = generate_random_input_file(path, x, n)
            #print("File category = ", file_category)
            #print("Filename = ", filename)

            if exact_file_index in previously_tested_file_indices:
                print("Repeat file detected, index = ", exact_file_index)
                pass

            elif exact_file_index not in previously_tested_file_indices:
                category_prediction = Single_File_Classification(input_path,
                                                               file_category,
                                                               rhythm_method,
                                                               spectral_centroid_method,
                                                                zero_crossing_rate_method)


                ###
                average_zcrs.append(average_zcr)
                sd_zcrs.append(sd_zcr)
                ###


                exact_prediction = check_if_correct_category(category_prediction, file_category)

                print("Prediction ", x)
                print("File category = ", file_category)
                prediction_correct += exact_prediction
                print("Number of predictions correct = ", prediction_correct, " of ", n_index)
                n_index += 1
                pass

            else:
                print("Error in detecting exact file index")
                pass
            
            pass
        







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
    exact_prediction = Single_File_Classification(input_path,
                                                   file_category,
                                                   rhythm_method,
                                                   spectral_centroid_method)
        













"""
# correlation between average and standard deviation spectral centroid values
pearson_correlation_value = scipy.stats.pearsonr(average_spectral_centroids, average_spectral_sds)
finds correlation in average and sd of spectral centroid
print("")
print("Pearson correlation value = ", pearson_correlation_value)
pearson_correlations.append(pearson_correlation_value)
"""
"""

#input_file = numpy.array(input_file[1], dtype = int)  # create numpy array of "original" input file
#matplotlib.pyplot.plot(averages_over_sds, label = file_category)  # plot input waveform

average_inverse_COV = sum(averages_over_sds)/len(averages_over_sds) # average of the inverse coefficients of variance!
print("Average inverse Coeff of variance = ", average_inverse_COV)

heatmap, xedges, yedges = numpy.histogram2d(average_spectral_centroids, average_spectral_sds, bins=20)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

matplotlib.pyplot.clf()
matplotlib.pyplot.imshow(heatmap.T, extent=extent, origin='lower')

#matplotlib.pyplot.plot(initial_peak_points,initial_peak_values, label= "peak_values")  # plot detected peak values
#pyplot.xlabel("File index in category")
#pyplot.ylabel("Average SC / Standard Dev. of SC")
#pyplot.legend()

#matplotlib.pyplot.plot(flattened_peak_points, flattened_peak_values, label= "flat_peak_values", marker="x")  # plot flattened peak values
#pyplot.legend()
print("Heatmap = ", heatmap)
print("X edges = ", xedges)
print("Y edges = ", yedges)
pyplot.show()

"""






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
