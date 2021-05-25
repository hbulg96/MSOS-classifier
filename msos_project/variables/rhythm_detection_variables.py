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



