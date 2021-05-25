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


filename = "7M5.wav"
category = "Music"
input_path = r"C:\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\\"
input_path = input_path + category + "\\" + filename
output_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Python DSP files\Output data files\Moving average filter outputs\\"

input_file = read(input_path)
print("input_file= ", input_file)
print(len(input_file))
initial_peak_points, initial_peak_values = peakdetection.peakdetection(input_file, window_length = 250000, crest_factor=10)

peak_file = numpy.array((initial_peak_points, initial_peak_values), dtype = int)

input_file = numpy.array(input_file[1], dtype = int)
matplotlib.pyplot.plot(input_file, label= "input_file")
matplotlib.pyplot.plot(initial_peak_points,initial_peak_values, label= "peak_values", marker="o")
pyplot.xlabel("Time")
pyplot.ylabel("Amplitude")
pyplot.legend()

flattened_peak_points, flattened_peak_values = peakflatten.peakflatten(peak_file, nearest_value_cutoff=2000)

matplotlib.pyplot.plot(flattened_peak_points, flattened_peak_values, label= "peak_values", marker="o")
pyplot.xlabel("Time")
pyplot.ylabel("Amplitude")
pyplot.legend()


pyplot.show()








