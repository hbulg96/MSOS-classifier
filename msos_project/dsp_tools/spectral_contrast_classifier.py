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



def spectral_contrast_classifier(input_file, show_graph=False):

    input_file = read(input_file)  # read wav file
    fs = input_file[0]
    input_file = numpy.array(input_file[1], dtype = float)  # interpret file as numpy array

    S = numpy.abs(librosa.stft(input_file))
    contrast = (librosa.feature.spectral_contrast(input_file, S=S, sr=fs, n_fft=2048))

    standard_deviation_of_contrast_bands = []
    
    for freq_band in contrast:
        band_standard_deviation = sum(freq_band)/len(freq_band)
        standard_deviation_of_contrast_bands.append(band_standard_deviation)
        pass

    if show_graph == True:
        for val in contrast:
            print("Contrast val = ", val)
            pass

        print("Standard deviation of contrast per freq. band = ", standard_deviation_of_contrast_bands)
        plt.figure()
        plt.subplot(2, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(S,
                                                         ref=numpy.max),
                                 y_axis='log')

        plt.colorbar(format='%+2.0f dB')
        plt.title('Power spectrogram')
        plt.subplot(2,1,2)
        librosa.display.specshow(contrast, x_axis='time')
        plt.colorbar()
        plt.ylabel('Frequency bands')
        plt.title('Spectral Contrast')
        plt.tight_layout()
        plt.show()
        pass

    elif show_graph == False:
        pass

    else:
        print("No show_graph variable in zero_crossing_rate_classifier")
        pass
    pass

    return(standard_deviation_of_contrast_bands)

