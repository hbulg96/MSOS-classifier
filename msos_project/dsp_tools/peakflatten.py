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


def peakflatten(input_peaks_file, nearest_value_cutoff= 20, peak_centre="max", all_positive=True):

    """
    Used to condense nearby samples into "single" peaks

    takes an input file which contains peak values against time
    
    nearest_value_cutoff represents the amount of samples that will be
    counted in a continuous window before creating a brand new one.
    
    peak_centre represents whether to use the "max" value,
    or the "centre" value of the detected peak window
    as the peak centre
    """
    #print("Input file = ", input_peaks_file)

    input_peaks_locations = numpy.array(input_peaks_file[0], dtype=int)
    #print("Input peak locations = ", input_peaks_locations)
    
    input_peaks_values = numpy.array(input_peaks_file[1], dtype = int)  # ensure file is numpy int array (removes upper and lower numerical limits)
    #print("Input peak values = ", input_peaks_values)
    

    """
    output_file = []  # initialise output file (if writing an output wav file)
    number_of_windows_required = int(round((len(input_peaks_locations)/window_length),0))  # number of required windows
    
    location_windows = numpy.array_split(input_peaks_locations, number_of_windows_required)  # array of window arrays of input file
    value_windows = numpy.array_split(input_peaks_values, number_of_windows_required)
    print("Peak nearest value cutoff = ", nearest_value_cutoff)
    print("Peak centre metric = ", peak_centre)

    print("SPLIT METHOD = ", location_windows)
    """

    location_windows = []  # array containing subwindows of location values, represeting a single peak to be flattened
    value_windows = []
    location_window = []  # initialise first subwindows holding peaks
    value_window = []
    
    #print("Length of input array = ", len(input_peaks_locations))
    
    for n in range(len(input_peaks_locations)):
        
        location_window.append(input_peaks_locations[n])
        value_window.append(input_peaks_values[n])
        try:
            if n == (len(input_peaks_locations)-1):
                difference_value = 0
                pass
            else:
                difference_value = (input_peaks_locations[n+1] - input_peaks_locations[n])
        except Exception as err:
            print(traceback.format_exc())
            difference_value = 0
            pass
        
            
        if difference_value >= nearest_value_cutoff:
            """
            If difference value is larger than cutoff, write current location_window
            into the larger location_windows array, and create a new location_window for the
            next iteration
            """
            location_window = numpy.array(location_window, dtype=int)  # convert subwindows to numpy arrays
            value_window = numpy.array(value_window, dtype=int)
            location_windows.append(location_window)  # append subwindows into larger windows
            value_windows.append(value_window)
            location_window = []  # initialise first subwindows holding peaks for next loop
            value_window = []
            pass

        elif difference_value < nearest_value_cutoff:
            pass

        else:
            print("Error in detecting difference value or cutoff value in PeakFlatten, see file")
            pass

    # final pass to close the last set of peak values
    location_window = numpy.array(location_window, dtype=int)  # convert subwindows to numpy arrays
    value_window = numpy.array(value_window, dtype=int)
    location_windows.append(location_window)  # append subwindows into larger windows
    value_windows.append(value_window)

    #print("Location windows = ", location_windows)
    

    try:

        x = 0  # used to index local maxima samples into input_peaks_file array
        peak_points = []  # initialise single peak value index
        peak_values = []  # initialise vector containing peak values (magnitude)

        for n in range(len(location_windows)):
            peak_value, peak_index = condensepeaks(location_windows[n], value_windows[n], peak_centre) # read single peak value and index from window
            peak_index = x + peak_index  # find peak index in larger input file
            
            for val in location_windows:
                #x +=1
                pass

            if all_positive == True:
                peak_value = abs(peak_value)
                pass
            elif all_positive == False:
                pass
            else:
                print("Error in detecting all_positive variable")
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

    #print("Window single peak value method = ", peak_centre_method)
    
    if peak_centre_method == "max":

        abs_val_array = []
        for n in range(len(input_window_values_array)):
            abs_val_array.append(abs(input_window_values_array[n]))
            pass
        abs_val_array = numpy.array(abs_val_array, dtype = int)
    
        window_max = numpy.max(abs_val_array)  # find max val in abs_val array
        window_max_index = numpy.argmax(abs_val_array)  # find location of value by indexing into window_max_index
        window_max = input_window_values_array[window_max_index]  # find "true" value (no abs()) of max by indexing into original array
        
        window_max_index = input_window_locations_array[window_max_index]  # find "true" index location in original array
        
        single_peak_value = window_max
        single_peak_index = window_max_index
        
        pass

    elif peak_centre_method == "centre":

        print("PLEASE")
        print("FIX THIS!!")
        print("NO CURRENT PEAK CENTRE METHOD APART FROM MAX")

        window_length = len(input_window_locations_array)

        window_centre_index = int(round(((float(window_length))/2), 0))
        window_centre = input_window_locations_array[window_centre_index]

        single_peak_value = window_centre
        single_peak_index = window_centre_index
        pass

    else:
        raise ValueError("ValueError incorrect peak_centre_method entered in peakflatten")
        pass

    
    #print("Window single peak value = ", single_peak_value)
    #print("Window single peak index in window = ", single_peak_index)

    return(single_peak_value, single_peak_index)
    pass

