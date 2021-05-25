"""
This archived version uses static "windows",
taking only a window length argument

Changed to nearest-value approach on 16/11/2020

"""



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


def peakflatten(input_peaks_file, window_length=100, peak_centre="max"):

    """
    Used to condense nearby samples into "single" peaks

    takes an input file which contains peak values against time
    
    window_length represents the amount of samples around the peak value
    which can be considered part of that particular peak.
    Consider making a different version that uses a numerical harshness
    based on the number of close surrounding values also
    
    peak_centre represents whether to use the "max" value,
    or the "centre" value of the detected peak window
    as the peak centre
    """
    print("Input file = ", input_peaks_file)

    input_peaks_locations = numpy.array(input_peaks_file[0], dtype=int)
    print("Input peak locations = ", input_peaks_locations)
    
    input_peaks_values = numpy.array(input_peaks_file[1], dtype = int)  # ensure file is numpy int array (removes upper and lower numerical limits)
    print("Input peak values = ", input_peaks_values)
    

    
    output_file = []  # initialise output file (if writing an output wav file)
    number_of_windows_required = int(round((len(input_peaks_locations)/window_length),0))  # number of required windows
    
    location_windows = numpy.array_split(input_peaks_locations, number_of_windows_required)  # array of window arrays of input file
    value_windows = numpy.array_split(input_peaks_values, number_of_windows_required)
    print("Peak window length = ", window_length)
    print("Peak centre metric = ", peak_centre)
    print("Locations windows = ", location_windows)
    print("Values in windows = ", value_windows)

    try:

        x = 0  # used to index local maxima samples into input_peaks_file array
        peak_points = []  # initialise single peak value index
        peak_values = []  # initialise vector containing peak values (magnitude)

        for n in range(len(location_windows)):
            peak_value, peak_index = condensepeaks(location_windows[n], value_windows[n], peak_centre) # read single peak value and index from window
            peak_index = x + peak_index  # find peak index in larger input file
            
            for val in location_windows:
                x +=1
                pass

            peak_points.append(peak_index)
            peak_values.append(peak_value)
            pass
        pass                  

    except Exception:
        print(traceback.format_exc())
        pass


    #print("Single peak locations found = ", peak_points)
    for val in peak_values:
        #print("Single peak values found = ", val)
        pass


    return(peak_points, peak_values)
    pass



def condensepeaks(input_window_locations_array, input_window_values_array, peak_centre_method):

    print("Window single peak value method = ", peak_centre_method)
    
    if peak_centre_method == "max":

        abs_val_array = []
        for n in range(len(input_window_values_array)):
            abs_val_array.append(abs(input_window_values_array[n]))
            pass
        abs_val_array = numpy.array(abs_val_array, dtype = int)
    
        window_max = numpy.max(abs_val_array)
        window_max_index = numpy.argmax(abs_val_array)
        window_max = input_window_values_array[window_max_index]
        window_max_index = input_window_locations_array[window_max_index]
        
        single_peak_value = window_max
        single_peak_index = window_max_index
        
        pass

    elif peak_centre_method == "centre":

        window_length = len(input_window_locations_array)

        window_centre_index = int(round(((float(window_length))/2), 0))
        window_centre = input_window_locations_array[window_centre_index]

        single_peak_value = window_centre
        single_peak_index = window_centre_index
        pass

    else:
        raise ValueError("ValueError incorrect peak_centre_method entered in peakflatten")
        pass

    
    print("Window single peak value = ", single_peak_value)
    print("Window single peak index in window = ", single_peak_index)

    return(single_peak_value, single_peak_index)
    pass

