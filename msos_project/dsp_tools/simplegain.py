"""
Remember!

This is an unbuffered moving average this is NOT symmetrical about the averaged sample, meaning
you will lose samples at the end of the file equal to the number of averages taken!
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

def simplegain(input_file, output_name, output_path, gain_value=1):
    """
    Simple gain boost

    gain_value is a value from 0 to 1
    1 represents the maximum value in the input value being boosted to the
    upper limit of the int16 file format (32767)
    """
    print("Filename = ", output_name)
    print("Gain value = ", gain_value)
    input_file = read(input_file)  # read wav file
    input_file = numpy.array(input_file[1], dtype = int)  # interpret file as numpy array
    # read audio samples of input wav file
    output_file = []  # initialize output file as an array

    try:

        max_value_in_file = numpy.amax(input_file)  # find the largest value in the input file
        max_gain_ratio = (float(32767) / float(max_value_in_file))  # find the multiplication ratio of the max value in file to max allowed value
        max_gain_ratio = (float(max_gain_ratio) * float(gain_value))  # multiply max allowed ratio to user specified gain ratio
        
        for input_sample in input_file:
            output_sample = (float(input_sample) * float(max_gain_ratio))  # apply gain boost to sample
            output_sample = int(round(output_sample, 0))  # round mean down to int. sample
            output_file.append(output_sample)  # add gain boosted value to output file
            pass

    except Exception as err:
        print(traceback.format_exc())
        pass

    output_file = numpy.array(output_file, dtype = int)
    output_file = (output_file).astype(numpy.int16) # convert output_file to numpy array to write to wav file
    output_path = output_path + output_name
    write(output_path, 44100, output_file)  # write output_file to output_path destination
    print("file written to ", str(output_path))
    return(output_file)
    pass


filename = "39L.wav"
input_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Python DSP files\Output data files\Moving average filter outputs\\"
input_path = input_path + filename

output_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Python DSP files\Output data files\Gain Boost outputs\\"
gain_boosted_file = simplegain(input_path, filename, output_path, gain_value=0.8)

"""
test_file = read(r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\Effects\0M8.wav")
test_file = numpy.array(test_file[1], dtype = int)
matplotlib.pyplot.plot(test_file)
pyplot.xlabel("Time")
pyplot.ylabel("Amplitude")
pyplot.show()
"""




matplotlib.pyplot.plot(gain_boosted_file)
pyplot.xlabel("Time")
pyplot.ylabel("Amplitude")
pyplot.show()



"""
f, t, Sxx = signal.spectrogram(average_effect_file, 44100)
pyplot.pcolormesh(t, f, Sxx, shading='gouraud')
pyplot.ylabel('Frequency [Hz]')
pyplot.xlabel('Time [sec]')
pyplot.show()
"""
        

