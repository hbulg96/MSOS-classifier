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
    print("Input filename = ", input_file)
    current_filename = str(path) + "\\" + input_file
    
    input_path = current_filename
    category = sound_category
    
    #print("Input file category = ", sound_category)
    return(input_path, category, exact_file_index, input_file)




def Single_File_Classification(input_path,
                               input_sound_category,
                               test_method,
                               rhythm_method=False,
                               spectral_centroid_method=False,
                               zero_crossing_rate_method=False,
                               rms_variation_method=False):

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
        average_zcr, sd_zcr, file_rms = zero_crossing_rate_classifier.zero_crossing_rate_classifier(input_path, show_graph=False)
        
        zcr_category_weights = [0,0,0,0,0]
        pass

    elif zero_crossing_rate_method == False:
        pass

    else:
        print("Error in detecting zero_crossing_rate_method")
        pass


    if rms_variation_method == True:
        rms_in_time, window_values = rms_variation_classifier.rms_variation_classifier(input_path,
                                                                 show_graph=False)
        
        pass

    elif rms_variation_method == False:
        pass

    else:
        print("Error in detecting rms_variation_method")
        pass

    if rms_skew_method == True:
        input_file = read(input_path)  # read wav file
        fs = input_file[0]
        input_file = numpy.array(input_file[1], dtype = float)  # interpret file as numpy array

        number_of_desired_hop_frames = int(round((len(input_file)/2000), 0))
        file_rms_over_time = (librosa.feature.rms(input_file, frame_length=4096, hop_length=number_of_desired_hop_frames))[0]
        feature_1 = scipy.stats.skew(file_rms_over_time, nan_policy='omit') * 1000
        feature_2 = numpy.std(file_rms_over_time)
        pass

    else:
        pass


    if test_method == True:
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

        # most important band (feature band)
        feature_band_index = max_contrast_band_index

        feature_band = feature_1[feature_band_index] # contrast band with the highest average contrast value,
        # representing the most interesting/intentional sound?

        # "least" important band (noise band)
        noise_band_index = min_contrast_band_index

        noise_band = feature_1[noise_band_index]

        max_contrast_all_bands = [] # location of the max spectral contrast at any time
        
        for value_index in range(length_of_contrast_values):
            # find index of current spectral contrast value
            contrast_values_per_band = []
            for freq_band in range(number_of_bands):
                # find max value in all bands
                current_band = feature_1[freq_band]
                contrast_values_per_band.append(current_band[value_index])
                pass

            max_contrast_value_band = max(contrast_values_per_band)
            mcvb_index = contrast_values_per_band.index(max_contrast_value_band)
            max_contrast_all_bands.append(mcvb_index)


        """5
        pyplot.figure(1)
        pyplot.imshow(feature_1, aspect='auto', origin="lower", cmap="coolwarm")
        pyplot.ylabel('Frequency Band')
        pyplot.xlabel('Time (DFT bin)')
        pyplot.title("Spectral Contrast")

        # add lines for feature band and noise band

        feature_band_x_points = [0, (feature_1.shape[1] - 1)]
        feature_band_y_points = [feature_band_index, feature_band_index]
        pyplot.plot(feature_band_x_points, feature_band_y_points, color='r',linewidth=3, label='feature band')

        noise_band_x_points = [0, (feature_1.shape[1] - 1)]
        noise_band_y_points = [noise_band_index, noise_band_index]
        pyplot.plot(noise_band_x_points, noise_band_y_points, color='b', linewidth=3, label='noise band')

        pyplot.plot(range(len(max_contrast_all_bands)), max_contrast_all_bands, color='g', label='max spectral contrast value')

        pyplot.legend()

        # plot feature and noise bands in their own graphs
        # feature band
        pyplot.figure(2)
        pyplot.plot(feature_band)
        pyplot.ylabel('Spectral Contrast')
        pyplot.xlabel('Time (DFT bins)')
        pyplot.title("Feature Band spectral contrast values")

        # noise band
        #pyplot.figure(3)
        #pyplot.plot(noise_band)
        pyplot.show()

        # broadband spectral contrast average
        average_broadband_contrast = 0
        for freq_band in range(number_of_bands):
            current_band = feature_1[freq_band]
            average_broadband_contrast += (sum(current_band)/len(current_band))
            pass
        average_broadband_contrast = average_broadband_contrast / number_of_bands  # broadband spectral contrast value
        """
        
        feature_1 = sum(feature_band)/len(feature_band) # feature band spectral contrast mean
        feature_2 = numpy.std(feature_band)/feature_1  # # coefficient of variation

    
        pass

    elif test_method == False:
        pass
    else:
        pass
    
    """
    category_weights = sc_category_weights
    print("Spectral centroid category weights = ", sc_category_weights)
    """
    #category_weights = zcr_category_weights
    # here, average other weights from the other DSP layers before prediction
    
    #category_prediction = CheckCategoryFromWeights(category_weights)
    #print("Category prediction = ", category_prediction)


    feature_1_name = "Feature band contrast mean"
    feature_2_name = "Feature band contrast coefficient of variation"


    return(feature_1, feature_2, feature_1_name, feature_2_name)



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
zero_crossing_rate_method = False
rms_variation_method = False
rms_skew_method = False
test_method = True
n_index = 0



if classification_method == "DSP":
    print("Rhythm detection method")

    previously_tested_file_indices = []

    all_heatmaps = []
    all_xedges = []
    all_yedges = []

    all_categories_feature_1 = []
    all_categories_feature_2 = []
    
    for n in range(5):
        print("category index = ", n)
        category_feature_1 = []
        category_feature_2 = []

        category_heatmap = []
        category_yedges = []
        category_xedges = []
        #n = 1  # to select one category only!


        
        for x in range(10):
            print("file index = ", x)
            input_path, file_category, exact_file_index, filename = generate_random_input_file(path, x, n)
            #print("File category = ", file_category)
            #print("Filename = ", filename)

            if exact_file_index in previously_tested_file_indices:
                print("Repeat file detected, index = ", exact_file_index)
                pass

            elif exact_file_index not in previously_tested_file_indices:
                feature_1, feature_2, feature_1_name, feature_2_name = Single_File_Classification(input_path,
                                                                                                  file_category,
                                                                                                  test_method)

                category_feature_1.append(feature_1)
                category_feature_2.append(feature_2)
                
                n_index += 1
                pass

            else:
                print("Error in detecting exact file index")
                pass
            
            pass

        all_categories_feature_1.append(category_feature_1)
        all_categories_feature_2.append(category_feature_2)



        
        # plot features individually for each category, and heatmaps for each category
        """
        print(str(feature_1_name), " = " , category_feature_1)
        print(str(feature_2_name), " = ", category_feature_2)

        pyplot.figure(1)
        pyplot.scatter(category_feature_1, category_feature_2)

        pyplot.xlabel(feature_1_name)
        pyplot.ylabel(feature_2_name)
        #pyplot.show()
        """
        
        heatmap, xedges, yedges = numpy.histogram2d(category_feature_1, category_feature_2, bins=20)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        category_heatmap.append(heatmap)
        category_xedges.append(xedges)        
        category_yedges.append(yedges)

        all_heatmaps.append(category_heatmap)
        all_xedges.append(category_xedges)
        all_yedges.append(category_yedges)
        
        #pyplot.figure(2)
        matplotlib.pyplot.clf()
        matplotlib.pyplot.imshow(heatmap.T, extent=extent, origin='lower')
        pyplot.xlabel(feature_1_name)
        pyplot.ylabel(feature_2_name)
        pyplot.show()

        matplotlib.pyplot.scatter(category_feature_1, category_feature_2)
        pyplot.xlabel(feature_1_name)
        pyplot.ylabel(feature_2_name)
        pyplot.show()

        
        print("Heatmap = ", heatmap)
        print("X edges = ", xedges)
        print("Y edges = ", yedges)
   
        
    print("All Heatmaps = ", all_heatmaps)
    print("All xedges = ", all_xedges)
    print("All yedges = ", all_yedges)

    # plot big scatter diagram with all categories combined

    for x in range(len(all_categories_feature_1)):

        category_feature_1 = all_categories_feature_1[x]
        category_feature_2 = all_categories_feature_2[x]
        
        pyplot.scatter(category_feature_1, category_feature_2)
        pass

    pyplot.legend(["Music", "Nature", "Urban", "Human", "Effects"])
    pyplot.xlabel(feature_1_name)
    pyplot.ylabel(feature_2_name)

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

    filename = "GC6.wav"
    file_category = "Nature"
    input_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\\"
    input_path = input_path + file_category + "\\" + filename
    output_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Python DSP files\Output data files\Moving average filter outputs\\"
    feature_1, feature_2, feature_1_name, feature_2_name = Single_File_Classification(input_path,
                                                                                        file_category,
                                                                                        test_method)
        













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
"""

"""
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


"""
# Test method : feature and noise band

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


        pyplot.figure(1)
        pyplot.imshow(feature_1, aspect='auto', origin="lower", cmap="coolwarm")
        pyplot.ylabel('Frequency Band')
        pyplot.xlabel('Time (DFT bin)')
        pyplot.title("Spectral Contrast")

        # add lines for feature band and noise band

        feature_band_x_points = [0, (feature_1.shape[1] - 1)]
        feature_band_y_points = [feature_band_index, feature_band_index]
        pyplot.plot(feature_band_x_points, feature_band_y_points, color='r',linewidth=3, label='feature band')

        noise_band_x_points = [0, (feature_1.shape[1] - 1)]
        noise_band_y_points = [noise_band_index, noise_band_index]
        pyplot.plot(noise_band_x_points, noise_band_y_points, color='b', linewidth=3, label='noise band')

        pyplot.plot(range(len(max_contrast_all_bands)), max_contrast_all_bands, color='g', label='max spectral contrast value')

        pyplot.legend()

        # plot feature and noise bands in their own graphs
        # feature band
        pyplot.figure(2)
        pyplot.plot(feature_band)
        pyplot.ylabel('Spectral Contrast')
        pyplot.xlabel('Time (DFT bins)')
        pyplot.title("Feature Band spectral contrast values")

        # noise band
        #pyplot.figure(3)
        #pyplot.plot(noise_band)
        pyplot.show()

        # broadband spectral contrast average
        average_broadband_contrast = 0
        for freq_band in range(number_of_bands):
            current_band = feature_1[freq_band]
            average_broadband_contrast += (sum(current_band)/len(current_band))
            pass
        average_broadband_contrast = average_broadband_contrast / number_of_bands  # broadband spectral contrast value


        


            
        feature_1 = time_spent_at_noise_band # Average of spectral contrast in all bands condensed into one value
        feature_2 = time_spent_at_feature_band  # amount of time ticks spent with max spentral contrast in the feature band

        #feature_1 = sum(feature_band)/len(feature_band) # feature band spectral contrast mean
        #feature_2 = numpy.std(feature_band)  # # standard dev


"""
