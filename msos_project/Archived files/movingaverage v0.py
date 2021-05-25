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

def movingaverage(input_file, output_name, output_path, number_of_points):
    
    input_file = read(input_file)
    input_file = numpy.array(input_file[1], dtype = int)
    # read audio samples of input wav file
    output_file = []

    try:
        
        for input_sample in range(len(input_file)):
            average_sample = 0
            for output_sample in range(number_of_points):
                average_sample += input_file[input_sample+output_sample]
                pass


            average_sample = average_sample / number_of_points
            average_sample = int(round(average_sample, 0))
            output_file.append(average_sample)
            pass

    except Exception as err:
        print(err)
        pass

    output_file = numpy.array(output_file, dtype = int)
    output_file = (output_file).astype(numpy.int16)
    output_name = output_name + ".wav"
    output_path = output_path + output_name
    write(output_path, 44100, output_file)
    print("file written to ", str(output_path))
    return(output_file)
    pass


filename = "2L6.wav"
input_path = r"C:\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\Effects\\"
input_path = input_path + filename

output_path = r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Python DSP files\Output data files\Moving average filter outputs\\"
moving_average_file = movingaverage(input_path, filename, output_path, number_of_points=10)

"""
test_file = read(r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\Effects\0M8.wav")
test_file = numpy.array(test_file[1], dtype = int)
matplotlib.pyplot.plot(test_file)
pyplot.xlabel("Time")
pyplot.ylabel("Amplitude")
pyplot.show()
"""




matplotlib.pyplot.plot(moving_average_file)
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
        

