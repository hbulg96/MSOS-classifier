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




def spectral_centroid_assign_weights(average_sc,
                                     standard_deviation_sc,
                                     all_heatmaps,
                                     all_xedges,
                                     all_yedges):
    """
    Now we will find the category weight value for each category, based on our probability heatmap
    """
    category_weights = []
    x_heatmap_index = -1
    y_heatmap_index = -1

    for classifier_index in range(len(all_heatmaps)):
        # for each classifier group in all_heatmaps

        x_category = all_xedges[classifier_index]
        y_category = all_yedges[classifier_index]
        
        for x_edge in range(len(x_category)-1):
            # for each x_edge value in the classifer group
            low_boundary = x_category[x_edge]
            high_boundary = x_category[x_edge+1]

            if average_sc >= low_boundary and average_sc <= high_boundary:
                x_heatmap_index = x_edge
                pass
            elif average_sc < low_boundary:
                pass  # if average lower than heatmap, very unlikely to be in category
            else:
                pass

        for y_edge in range(len(y_category)-1):
            low_boundary = y_category[y_edge]
            high_boundary = y_category[y_edge+1]

            if standard_deviation_sc >= low_boundary and standard_deviation_sc <= high_boundary:
                y_heatmap_index = y_edge
                pass
            elif standard_deviation_sc < low_boundary:
                pass
            else:
                pass

        if x_heatmap_index >= 0 and y_heatmap_index >= 0:
            category_weight = ((all_heatmaps[classifier_index])[x_heatmap_index])[y_heatmap_index]
            pass

        else:
            category_weight = 0
            pass
        
        category_weights.append(category_weight)


    return(category_weights)
