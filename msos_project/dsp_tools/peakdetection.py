"""
08/11/2020
added some code comments and "print("crest factor = ... "

25 seems to be a good crest factor for detecting large impulses
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
import math

def peakdetection(input_file, window_length, crest_factor):

    """
    Simple peak detection filter that does use RMS

    crest factor value is measured against the average of absolute values in the window
    """
    input_file = numpy.array(input_file[1], dtype = numpy.float64)

    #print("input file = ", input_file)
    # read audio samples of input wav file
    output_file = []
    number_of_windows_required = int(round((len(input_file)/window_length),0))
    windows = numpy.array_split(input_file, number_of_windows_required)
    #print("number of windows required = ", number_of_windows_required)
    #print("Crest factor = ", crest_factor)

    try:

        x = 0  # used to index local maxima samples into input_file array
        peak_points = []
        peak_values = []
        for window in windows:
            window_sum = 0
            for val in window:
                window_sum += ((val)**2)
                pass
            
            window_average = math.sqrt((window_sum)/len(window))  # average of values in the window
            #print("window average = ", window_average)
            peak_limit = abs(float(window_average) * float(crest_factor))  # limit that defines a local peak (against the window average)
            #print("peak limit = ", peak_limit)

            for sample in window:
                if abs(sample) > peak_limit:
                    peak_points.append(x)
                    peak_values.append(sample)
                    pass

                elif abs(sample) <= peak_limit:
                    pass

                else:
                    print("Error in detecting sample value or peak_limit value")
                    pass

                x = x+1
                pass

            pass
        #print("Sample positions array length = ", len(peak_points))
        #print("Sample magnitude array length = ", len(peak_values))
                    

    except Exception as err:
        print(err)
        print(traceback.format_exc())
        pass

    #print("Peak locations found = ", peak_points)
    #print("Peak values found = ", peak_values)
    
    
    return(peak_points, peak_values)
    pass



