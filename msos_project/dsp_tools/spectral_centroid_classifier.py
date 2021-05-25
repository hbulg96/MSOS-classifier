
import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import os
import timeit
import traceback
import msos_project
import librosa
from librosa import *
from librosa import display



def spectral_centroid_classifier(input_file, show_graph=False):

    input_file = read(input_file)  # read wav file
    fs = input_file[0]
    input_file = numpy.array(input_file[1], dtype = float)  # interpret file as numpy array

    output_file = librosa.feature.spectral.spectral_centroid(input_file, fs, n_fft=2000)

    output_file = output_file[0]
    # average spectral centroid across file
    average_spectral_centroid = sum(output_file)/len(output_file)
    # standard deviation of spectral centroid
    sd_spectral_centroid = numpy.std(output_file)
    
    if show_graph == True:
        print("Average spectral centroid = ", average_spectral_centroid)
        print("Standard deviation of spectral centroid = ", sd_spectral_centroid)
        S, phase = librosa.magphase(librosa.stft(y=input_file))

        times = librosa.times_like(output_file)
        fig, ax = plt.subplots()
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=numpy.max),
                                 y_axis='log', x_axis='time', ax=ax)
        ax.plot(times, output_file.T, label='Spectral centroid', color='w')
        ax.legend(loc='upper right')
        ax.set(title='log Power spectrogram')
        plt.show()

        pass

    elif show_graph == False:
        pass
    else:
        print("Error in detecting show_graph variable")
        pass

    return(average_spectral_centroid, sd_spectral_centroid)




"""
test_file = read(r"C:\\Users\h_bul\Documents\Acoustics Year 3\Project\Audio Resources\Development\Effects\0M8.wav")
test_file = numpy.array(test_file[1], dtype = int)
matplotlib.pyplot.plot(test_file)
pyplot.xlabel("Time")
pyplot.ylabel("Amplitude")
pyplot.show()
"""

"""
matplotlib.pyplot.plot(gain_boosted_file)
pyplot.xlabel("Time")
pyplot.ylabel("Amplitude")
pyplot.show()
"""
"""
f, t, Sxx = signal.spectrogram(average_effect_file, 44100)
pyplot.pcolormesh(t, f, Sxx, shading='gouraud')
pyplot.ylabel('Frequency [Hz]')
pyplot.xlabel('Time [sec]')
pyplot.show()
"""
        

