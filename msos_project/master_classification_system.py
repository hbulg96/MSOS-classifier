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
import msos_project.dsp_tools.spectral_centroid_classifier as spectral_centroid_classifier
import msos_project.dsp_tools.spectral_centroid_assign_weights as spectral_centroid_assign_weights
import msos_project.dsp_tools.zero_crossing_rate_classifier as zero_crossing_rate_classifier
import msos_project.dsp_tools.zero_crossing_rate_assign_weights as zero_crossing_rate_assign_weights
import msos_project.dsp_tools.rms_skew_classifier as rms_skew_classifier
import msos_project.dsp_tools.rms_skew_assign_weights as rms_skew_assign_weights
import msos_project.dsp_tools.spectral_contrast_feature_max_classifier as spectral_contrast_feature_max_classifier
import msos_project.dsp_tools.effective_duration_classifier as effective_duration_classifier

import msos_project.dsp_tools.assign_weights as assign_weights

import msos_project.dsp_tools.spectral_contrast_mean_classifier as spectral_contrast_mean_classifier
import msos_project.dsp_tools.note_onset_peak_classifier as note_onset_peak_classifier
from scipy import stats
from numpy import polyfit
import librosa
import time


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
                               zero_crossing_rate_method,
                               rms_skew_method,
                               spectral_contrast_feature_max_method,
                               spectral_contrast_mean_method,
                               note_onset_peak_method,
                               effective_duration_method):



    input_weights = []
    
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


        #print("Average spectral centroid = ", average_sc)
        #print("Standard deviation of spectral centroid = ", standard_deviation_sc)
        
        sc_category_weights = spectral_centroid_assign_weights.spectral_centroid_assign_weights(average_sc,
                                                                                            standard_deviation_sc,
                                                                                            all_heatmaps,
                                                                                            all_xedges,
                                                                                            all_yedges)


        #print("Average spectral centroid = ", average_sc)
        #print("Standard deviation of spectral centroid = ", standard_deviation_sc)
        input_weights.append(sc_category_weights)
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

        import msos_project.variables.ZCR_heatmap_variables  as ZCR_heatmap_vars

        average_zcr, sd_zcr, file_rms = zero_crossing_rate_classifier.zero_crossing_rate_classifier(input_path, show_graph=False)

        all_yedges = ZCR_heatmap_vars.all_yedges
        all_xedges = ZCR_heatmap_vars.all_xedges
        all_heatmaps = ZCR_heatmap_vars.all_heatmaps

        zcr_category_weights = zero_crossing_rate_assign_weights.zero_crossing_rate_assign_weights(average_zcr,
                                                                                                   sd_zcr,
                                                                                                   file_rms,
                                                                                                   all_heatmaps,
                                                                                                   all_xedges,
                                                                                                   all_yedges)
        input_weights.append(zcr_category_weights)
        pass

    elif zero_crossing_rate_method == False:
        pass

    else:
        print("Error in detecting zero_crossing_rate_method")
        pass


    if rms_skew_method == True:

        import msos_project.variables.RMS_skew_sd_heatmap_variables as RMS_skew_heatmap_vars

        # variables for graphing specific points in the RMS scatter dataset
        # essentially these are bounding conditions for the all categories scatter graph
        sd_low_lim = 0
        skew_low_lim = -5
        sd_hi_lim = 4000
        skew_hi_lim = -2

        rms_skew, rms_sd = rms_skew_classifier.rms_skew_classifier(input_path, show_graph=False,
                                                                   sd_low_lim=sd_low_lim, skew_low_lim=skew_low_lim,
                                                                   sd_hi_lim=sd_hi_lim, skew_hi_lim=skew_hi_lim)

        rms_yedges = RMS_skew_heatmap_vars.all_yedges
        rms_xedges = RMS_skew_heatmap_vars.all_xedges
        rms_heatmaps = RMS_skew_heatmap_vars.all_heatmaps

        rms_skew_weights = rms_skew_assign_weights.rms_skew_assign_weights(rms_skew,
                                                                           rms_sd,
                                                                           rms_heatmaps,
                                                                           rms_xedges,
                                                                           rms_yedges)

        input_weights.append(rms_skew_weights)
        pass

    elif rms_skew_method == False:
        pass

    else:
        print("Error in detecting rms_skew_method")
        pass

    if spectral_contrast_feature_max_method == True:

        import msos_project.variables.spectral_contrast_feature_max_heatmap_variables as sc_fm_heatmap_vars

        noise_min_metric, feature_max_metric = spectral_contrast_feature_max_classifier.spectral_contrast_feature_max_classifier(input_path,
                                                                                                                                 show_graph=False)


        print("Noise min metric = ", noise_min_metric)
        print("Feature max metric = ", feature_max_metric)

        sc_fm_yedges = sc_fm_heatmap_vars.all_yedges
        sc_fm_xedges = sc_fm_heatmap_vars.all_xedges
        sc_fm_heatmaps = sc_fm_heatmap_vars.all_heatmaps

        sc_fm_weights = assign_weights.assign_weights(feature_max_metric,
                                                      noise_min_metric,
                                                      sc_fm_heatmaps,
                                                      sc_fm_xedges,
                                                      sc_fm_yedges)
        

        input_weights.append(sc_fm_weights)

        

        pass

    elif spectral_contrast_feature_max_method == False:
        pass
    else:
        print("Error in detecting spectral_contrast_feature_max_method")
        pass

    if spectral_contrast_mean_method == True:

        import msos_project.variables.spectral_contrast_mean_heatmap_variables as sc_mean_cov_heatmap_vars

        feature_band_mean, feature_band_coeff_of_variation = spectral_contrast_mean_classifier.spectral_contrast_mean_classifier(input_path,
                                                                                                                                 show_graph=False)

        sc_mean_cov_yedges = sc_mean_cov_heatmap_vars.all_yedges
        sc_mean_cov_xedges = sc_mean_cov_heatmap_vars.all_xedges
        sc_mean_cov_heatmaps = sc_mean_cov_heatmap_vars.all_heatmaps

        sc_mean_cov_weights = assign_weights.assign_weights(feature_band_coeff_of_variation,
                                                      feature_band_mean,
                                                      sc_mean_cov_heatmaps,
                                                      sc_mean_cov_xedges,
                                                      sc_mean_cov_yedges)

        input_weights.append(sc_mean_cov_weights)
        pass
    elif spectral_contrast_mean_method == False:

        pass

    else:
        print("Error in detecting spectral_contrast_mean_method")
        pass


    if note_onset_peak_method == True:
        import msos_project.variables.note_onset_peak_heatmap_variables as note_onset_peak_heatmap_variables
        
        sd_of_peak_crest_factors, mean_of_peak_crest_factors = note_onset_peak_classifier.note_onset_peak_classifier(input_path,
                                                                                                                     show_graph=False)

        no_peak_cf_yedges = note_onset_peak_heatmap_variables.all_yedges
        no_peak_cf_xedges = note_onset_peak_heatmap_variables.all_xedges
        no_peak_cf_heatmaps = note_onset_peak_heatmap_variables.all_heatmaps

        no_peak_cf_weights = assign_weights.assign_weights(mean_of_peak_crest_factors,
                                                           sd_of_peak_crest_factors,
                                                           no_peak_cf_heatmaps,
                                                           no_peak_cf_xedges,
                                                           no_peak_cf_yedges)
        input_weights.append(no_peak_cf_weights)
        pass

    elif note_onset_peak_method == False:
        pass

    else:
        print("Error in detecting note_onset_peak_method")
        pass



    if effective_duration_method == True:
        import msos_project.variables.effective_duration_heatmap_variables as effective_duration_heatmap_variables

        effective_duration, log_attack_time  = effective_duration_classifier.effective_duration_classifier(input_path,
                                                                                                           show_graph=False)
        ed_heatmaps = effective_duration_heatmap_variables.all_heatmaps
        ed_xedges = effective_duration_heatmap_variables.all_xedges
        ed_yedges = effective_duration_heatmap_variables.all_yedges

        effective_duration_weights = assign_weights.assign_weights(log_attack_time,
                                                                   effective_duration,
                                                                   ed_heatmaps,
                                                                   ed_xedges,
                                                                   ed_yedges)

        input_weights.append(effective_duration_weights)
        pass

    elif effective_duration_method == False:
        pass

    else:
        print("Error in detecting effective_duration_method")
        pass


    if input_weights == []:
        input_weights = [[0, 0, 0, 0, 0]]
        pass

    else:
        pass

    
    category_weights = CombineWeights(input_weights)

    category_prediction = CheckCategoryFromWeights(category_weights)
    print("Category prediction = ", category_prediction)

    return(category_prediction)





def CombineWeights(input_weights):

    # combine input weights sensibly
    # return the combined weight

    output_weights = []

    # simple averaging method
    for weights_index in range(len(input_weights)):
        current_weight = input_weights[weights_index]

        
    for n in range(len(input_weights[0])):

        #print("N = ", n)  # weight number ([0, 1, 2, 3, 4]) for each category

        output_weight = []
        
        for weights_index in range(len(input_weights)):
            # current test method weights
            current_weight = input_weights[weights_index]
            #print("Current weight = ", current_weight)
            #print("category weight = ", current_weight[n])
            # weight number index for current test method (i.e. spectral centroid music weight [0.34])
            output_weight.append(current_weight[n])
            pass

        output_weight = sum(output_weight)/len(output_weight)  # simple average output weight list
        
        output_weights.append(output_weight)  # add single category weight to five length output weights list
        pass
    
    

    #print("Output weights = ", output_weights)        

    return(output_weights)



def CheckCategoryFromWeights(input_category_weights):

    print("Final Category Weights = ", input_category_weights)

    if input_category_weights == [0, 0, 0, 0, 0]:
        category_index = numpy.random.randint(0,5)
        print("Category weights 0, prediction randomised")
        # if all weights are 0, randomize the prediction
        pass
    else:
        category_index = input_category_weights.index(max(input_category_weights))
        pass

    sound_categories = ["Music", "Nature", "Urban", "Human", "Effects"] 
    category_prediction = sound_categories[category_index]

    return(category_prediction)



def NumberOfCorrectGuesses(file_category, exact_prediction):
    global category_guesses, correct_category_guesses
    
    file_categories = ["Music", "Nature", "Urban", "Human", "Effects"]

    file_category_index = file_categories.index(file_category)
    category_guesses[file_category_index] = (category_guesses[file_category_index]) + 1
    correct_category_guesses[file_category_index] = (correct_category_guesses[file_category_index]) + exact_prediction

    return(category_guesses, correct_category_guesses)



category_guesses = [0, 0, 0, 0, 0]
correct_category_guesses = [0, 0, 0, 0, 0]

prediction_correct = 0

# development path
path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\\"
# evaluation path
#path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Evaluation\Evaluation\\"

classification_method = "DSP"
rhythm_method = False
spectral_centroid_method = True
zero_crossing_rate_method = True
rms_skew_method = True
spectral_contrast_feature_max_method = True
spectral_contrast_mean_method = True
note_onset_peak_method = True
effective_duration_method = True
n_index = 0

average_timer = []

if classification_method == "DSP":

    previously_tested_file_indices = []
    
    for n in range(1):
        print("category index = ", n)
        #average_zcrs = []
        #sd_zcrs = []
        n = 4
        for x in range(20):
            print("file index = ", x)
            input_path, file_category, exact_file_index, filename = generate_random_input_file(path, x, n)
            #print("File category = ", file_category)
            print("Filename = ", filename)

            if exact_file_index in previously_tested_file_indices:
                print("Repeat file detected, index = ", exact_file_index)
                pass

            elif exact_file_index not in previously_tested_file_indices:

                tic = time.perf_counter()
                
                category_prediction = Single_File_Classification(input_path,
                                                               file_category,
                                                               rhythm_method,
                                                               spectral_centroid_method,
                                                                zero_crossing_rate_method,
                                                                 rms_skew_method,
                                                                 spectral_contrast_feature_max_method,
                                                                 spectral_contrast_mean_method,
                                                                 note_onset_peak_method,
                                                                 effective_duration_method)

                toc = time.perf_counter()
                print("Single file classification time = ", (toc-tic), " seconds")

                average_timer.append((toc-tic))

                exact_prediction = check_if_correct_category(category_prediction, file_category)

                print("Prediction ", x)
                print("File category = ", file_category)
                prediction_correct += exact_prediction
                print("Number of predictions correct = ", prediction_correct, " of ", n_index+1)
                
                category_guesses, correct_category_guesses = NumberOfCorrectGuesses(category_prediction, exact_prediction)

                
                n_index += 1
                pass

            else:
                print("Error in detecting exact file index")
                pass
            
            pass

        pass
    print("Categories = Music Nature Urban Human Effects")
    print("Category guesses = ", category_guesses)
    print("Correct category guesses = ", correct_category_guesses)

    average_timer = sum(average_timer)/len(average_timer)
    print("Average timer = ", average_timer)
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
                                                   spectral_centroid_method,
                                                  zero_crossing_rate_method,
                                                  rms_skew_method,
                                                  spectral_contrast_feature_max_method,
                                                  spectral_contrast_mean_method,
                                                  note_onset_peak_method,
                                                  effective_duration_method)
        













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
