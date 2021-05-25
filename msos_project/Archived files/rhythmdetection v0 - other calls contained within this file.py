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
import msos_project.dsp_tools.find_similar_magnitude_peaks as find_similar_magnitude_peaks


def rhythmdetection(input_file, peak_detection_window_length=250000, input_crest_factor=10, value_cutoff=2000, peak_style="max"):
    
    initial_peak_points, initial_peak_values = peakdetection.peakdetection(input_file, window_length = peak_detection_window_length, crest_factor=input_crest_factor)  
    peak_file = numpy.array((initial_peak_points, initial_peak_values), dtype = int) # create numpy array of peak points

    input_file = numpy.array(input_file[1], dtype = int)  # create numpy array of "original" input file
    matplotlib.pyplot.plot(input_file, label= "input_file")  # plot input waveform
    matplotlib.pyplot.plot(initial_peak_points,initial_peak_values, label= "peak_values", marker="o")  # plot detected peak values
    pyplot.xlabel("Time")
    pyplot.ylabel("Amplitude")
    pyplot.legend()

    flattened_peak_points, flattened_peak_values = peakflatten.peakflatten(peak_file, nearest_value_cutoff=value_cutoff, peak_centre=peak_style)
    # flatten peaks into single points

    #similar_magnitude_locations, similar_magnitude_values = find_similar_magnitude_peaks.find_similar_magnitude_peaks(flattened_peak_points, flattened_peak_values, magnitude_similarity=0.6)
    
    matplotlib.pyplot.plot(flattened_peak_points, flattened_peak_values, label= "peak_values", marker="o")  # plot flattened peak values
    pyplot.xlabel("Time")
    pyplot.ylabel("Amplitude")
    pyplot.legend()
    pyplot.show()

    # RHYTHMIC PROPERTIES OF THE PEAKS HERE

    

    # first, how far apart are the peak_points?
    # are all of the peak_points a similar distance apart?
    # are peak values of a similar magnitude a similar distance apart?
    
    # return values that are indicative of rhythm for instance:
    # average time between peaks
    # how close is the mean difference to each difference (variation)
    # are there different sets of peak magnitudes?

    return(flattened_peak_points, flattened_peak_values)



def peak_average_distance():

    pass

def peak_distance_similarity_to_average():

    pass
    





