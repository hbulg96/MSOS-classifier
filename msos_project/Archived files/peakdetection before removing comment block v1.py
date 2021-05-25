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

def peakdetection(input_file, window_length, crest_factor):

    """
    Simple peak detection filter that does NOT use RMS, simply converts
    negative values using python's inbuilt abs() function

    crest factor value is measured against the average of absolute values in the window
    """
    
    input_file = numpy.array(input_file[1], dtype = int)
    # read audio samples of input wav file
    output_file = []
    number_of_windows_required = int(round((len(input_file)/window_length),0))
    windows = numpy.array_split(input_file, number_of_windows_required)
    print("number of windows required = ", number_of_windows_required)
    print("Crest factor = ", crest_factor)

    try:

        x = 0  # used to index local maxima samples into input_file array
        peak_points = []
        peak_values = []
        for window in windows:
            window_sum = 0
            for val in window:
                window_sum += abs(val)
                pass
            
            window_average = (window_sum)/len(window)  # average of values in the window
            #print("window average = ", window_average)
            peak_limit = abs(float(window_average) * float(crest_factor))  # limit that defines a local peak (against the window average)
            #print("peak limit = ", peak_limit)

            for sample in window:
                if abs(sample) > peak_limit:
                    peak_points.append(x)
                    peak_values.append(sample)
                    #print("Peak point found at sample = ", x)  # why is this printing every value?
                    #print("Sample value = ", sample)
                    pass

                elif abs(sample) <= peak_limit:
                    pass

                else:
                    print("Error in detecting sample value or peak_limit value")
                    pass

                x += 1
                pass
                    

    except Exception as err:
        print(err)
        print(traceback.format_exc())
        pass

    print("Peak locations found = ", peak_points)
    print("Peak values found = ", peak_values)
    matplotlib.pyplot.plot(input_file, label= "input_file")
    matplotlib.pyplot.plot(peak_points,peak_values, label= "peak_values", marker="o")
    #pyplot.xlabel("Time")
    #pyplot.ylabel("Amplitude")
    pyplot.legend()
    pyplot.show()
    return(peak_points, peak_values)
    pass




"""
filename = "6EC.wav"
category = "Music"
input_path = r"C:\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\\"
input_path = input_path + category + "\\" + filename

output_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Python DSP files\Output data files\Moving average filter outputs\\"
peak_points, peak_values = peakdetection(input_path, window_length = 250000, crest_factor=5)

test_file = read(r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\Effects\0M8.wav")
test_file = numpy.array(test_file[1], dtype = int)
matplotlib.pyplot.plot(test_file)
pyplot.xlabel("Time")
pyplot.ylabel("Amplitude")
pyplot.show()

f, t, Sxx = signal.spectrogram(average_effect_file, 44100)
pyplot.pcolormesh(t, f, Sxx, shading='gouraud')
pyplot.ylabel('Frequency [Hz]')
pyplot.xlabel('Time [sec]')
pyplot.show()
"""
        
