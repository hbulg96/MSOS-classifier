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
import msos_project.dsp_tools.rhythmdetection as rhythmdetection
import msos_project.dsp_tools.find_similar_magnitude_peaks as find_similar_magnitude_peaks
import msos_project.dsp_tools.find_rhythmic_packets as find_rhythmic_packets
from numpy import random
import msos_project.classification_1_rhythm_time_domain_v0_standalone as classifier1
import msos_project.dsp_tools.spectral_centroid_classifier as spectral_centroid_classifier
from scipy import stats
from numpy import polyfit


all_xedges =  [[  54.  ,  151.55,  249.1 ,  346.65,  444.2 ,  541.75,  639.3 ,
        736.85,  834.4 ,  931.95, 1029.5 , 1127.05, 1224.6 , 1322.15,
       1419.7 , 1517.25, 1614.8 , 1712.35, 1809.9 , 1907.45, 2005.  ], [  32.  ,  130.65,  229.3 ,  327.95,  426.6 ,  525.25,  623.9 ,
        722.55,  821.2 ,  919.85, 1018.5 , 1117.15, 1215.8 , 1314.45,
       1413.1 , 1511.75, 1610.4 , 1709.05, 1807.7 , 1906.35, 2005.  ], [  32.  ,  130.65,  229.3 ,  327.95,  426.6 ,  525.25,  623.9 ,
        722.55,  821.2 ,  919.85, 1018.5 , 1117.15, 1215.8 , 1314.45,
       1413.1 , 1511.75, 1610.4 , 1709.05, 1807.7 , 1906.35, 2005.  ], [  29. ,  127.8,  226.6,  325.4,  424.2,  523. ,  621.8,  720.6,
        819.4,  918.2, 1017. , 1115.8, 1214.6, 1313.4, 1412.2, 1511. ,
       1609.8, 1708.6, 1807.4, 1906.2, 2005. ], [  32.  ,  126.05,  220.1 ,  314.15,  408.2 ,  502.25,  596.3 ,
        690.35,  784.4 ,  878.45,  972.5 , 1066.55, 1160.6 , 1254.65,
       1348.7 , 1442.75, 1536.8 , 1630.85, 1724.9 , 1818.95, 1913.]]


all_yedges = [[0.47712125, 0.61730549, 0.75748972, 0.89767396, 1.03785819,
       1.17804242, 1.31822666, 1.45841089, 1.59859512, 1.73877936,
       1.87896359, 2.01914783, 2.15933206, 2.29951629, 2.43970053,
       2.57988476, 2.72006899, 2.86025323, 3.00043746, 3.14062169,
       3.28080593], [0.30103   , 0.45060238, 0.60017476, 0.74974714, 0.89931952,
       1.0488919 , 1.19846428, 1.34803665, 1.49760903, 1.64718141,
       1.79675379, 1.94632617, 2.09589855, 2.24547093, 2.39504331,
       2.54461569, 2.69418807, 2.84376045, 2.99333283, 3.14290521,
       3.29247759], [0.47712125, 0.61754306, 0.75796486, 0.89838666, 1.03880847,
       1.17923027, 1.31965207, 1.46007387, 1.60049568, 1.74091748,
       1.88133928, 2.02176108, 2.16218289, 2.30260469, 2.44302649,
       2.5834483 , 2.7238701 , 2.8642919 , 3.0047137 , 3.14513551,
       3.28555731], [0.60205999, 0.73586036, 0.86966073, 1.00346109, 1.13726146,
       1.27106183, 1.40486219, 1.53866256, 1.67246293, 1.80626329,
       1.94006366, 2.07386403, 2.2076644 , 2.34146476, 2.47526513,
       2.6090655 , 2.74286586, 2.87666623, 3.0104666 , 3.14426696,
       3.27806733], [0.        , 0.16293186, 0.32586373, 0.48879559, 0.65172746,
       0.81465932, 0.97759118, 1.14052305, 1.30345491, 1.46638678,
       1.62931864, 1.79225051, 1.95518237, 2.11811423, 2.2810461 ,
       2.44397796, 2.60690983, 2.76984169, 2.93277355, 3.09570542,
       3.25863728]]

all_heatmaps = [[[0., 0., 0., 1., 2., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        2., 1., 1., 0.],
       [0., 0., 1., 0., 0., 0., 2., 0., 1., 1., 2., 1., 0., 2., 1., 3.,
        5., 4., 3., 3.],
       [0., 0., 0., 0., 1., 0., 2., 1., 0., 1., 0., 1., 1., 2., 3., 1.,
        1., 4., 5., 2.],
       [1., 0., 0., 0., 0., 1., 2., 1., 1., 0., 1., 0., 2., 1., 2., 1.,
        0., 6., 7., 5.],
       [1., 0., 0., 0., 0., 0., 0., 3., 0., 0., 1., 0., 0., 2., 0., 2.,
        4., 2., 0., 5.],
       [0., 0., 0., 0., 0., 0., 1., 3., 0., 0., 1., 0., 1., 0., 3., 0.,
        4., 1., 2., 4.],
       [0., 0., 0., 0., 0., 1., 1., 2., 0., 0., 0., 1., 1., 1., 1., 3.,
        3., 1., 2., 2.],
       [0., 0., 0., 0., 0., 0., 1., 2., 2., 1., 1., 0., 1., 0., 1., 0.,
        0., 5., 1., 1.],
       [0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 1., 0., 1., 2., 1., 0.,
        1., 2., 2., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 2.,
        1., 1., 1., 2.],
       [0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        1., 0., 1., 1.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 2., 0., 1., 1.,
        0., 1., 1., 3.],
       [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 2., 0.,
        0., 3., 0., 0.],
       [0., 0., 0., 0., 0., 1., 1., 2., 1., 0., 0., 1., 0., 0., 1., 0.,
        2., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 2., 0.,
        1., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.,
        0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 1., 5.,
        0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 1., 2.,
        1., 2., 2., 0.],
       [0., 0., 0., 0., 0., 1., 1., 0., 3., 0., 1., 0., 4., 4., 1., 1.,
        1., 0., 1., 0.],
       [0., 0., 0., 1., 1., 0., 3., 1., 1., 1., 2., 2., 2., 3., 1., 1.,
        0., 2., 2., 1.]], [[1., 0., 1., 3., 1., 0., 3., 3., 5., 0., 0., 1., 0., 2., 0., 3.,
        4., 4., 5., 1.],
       [0., 0., 0., 0., 1., 1., 1., 2., 2., 1., 0., 2., 2., 4., 3., 9.,
        3., 6., 6., 4.],
       [0., 0., 0., 0., 0., 0., 0., 0., 2., 1., 4., 1., 2., 3., 1., 0.,
        4., 7., 4., 4.],
       [0., 0., 0., 0., 1., 0., 1., 1., 1., 2., 5., 4., 2., 5., 2., 4.,
        5., 3., 5., 2.],
       [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 2., 4., 2., 4., 0.,
        2., 4., 0., 3.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 2., 1., 1., 1.,
        2., 1., 3., 4.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 2., 2., 0.,
        4., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0.,
        1., 2., 2., 3.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 2., 2.,
        1., 1., 2., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 2., 0., 0., 0., 0.,
        2., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1.,
        1., 0., 2., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
        0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 2.,
        0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        2., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 2.,
        0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
        0., 0., 1., 1.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1., 1., 2.,
        0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1.,
        0., 0., 0., 3.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 2., 0., 2., 1., 2., 2., 5., 2., 1.,
        1., 1., 0., 1.]], [[1., 0., 1., 2., 4., 3., 6., 7., 0., 1., 1., 1., 3., 4., 1., 2.,
        1., 1., 2., 3.],
       [1., 0., 0., 1., 2., 1., 2., 5., 3., 0., 0., 1., 1., 2., 2., 0.,
        1., 3., 1., 3.],
       [0., 0., 1., 0., 1., 1., 2., 1., 1., 0., 4., 1., 0., 3., 2., 2.,
        1., 5., 2., 2.],
       [0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 3., 2., 1., 0., 0., 1.,
        1., 0., 2., 3.],
       [0., 0., 0., 1., 1., 0., 0., 2., 0., 2., 0., 0., 0., 1., 0., 0.,
        1., 1., 1., 1.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 2., 1., 1., 2., 0., 1.,
        3., 2., 3., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 2., 0., 0., 1., 0., 0.,
        1., 2., 1., 1.],
       [0., 0., 0., 0., 0., 1., 0., 2., 0., 1., 0., 0., 0., 0., 3., 2.,
        3., 4., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 2., 0., 1., 1., 0., 3., 0.,
        2., 0., 0., 1.],
       [0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 2.,
        2., 1., 1., 2.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 1., 0., 0.,
        1., 0., 1., 1.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 2., 0., 1.,
        0., 2., 2., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
        2., 1., 2., 2.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0.,
        0., 2., 2., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0.,
        1., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
        0., 0., 1., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        1., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
        1., 0., 0., 1.],
       [0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
        1., 0., 1., 3.],
       [0., 0., 0., 1., 0., 5., 9., 5., 2., 1., 2., 2., 2., 2., 3., 4.,
        1., 3., 2., 3.]], [[1., 0., 3., 3., 7., 7., 7., 6., 4., 3., 4., 2., 1., 0., 0., 2.,
        4., 4., 4., 5.],
       [0., 1., 0., 1., 0., 2., 4., 1., 2., 3., 6., 3., 5., 1., 1., 4.,
        4., 7., 8., 6.],
       [1., 0., 1., 0., 0., 1., 3., 0., 2., 4., 2., 5., 6., 3., 4., 2.,
        4., 5., 4., 2.],
       [0., 0., 1., 0., 0., 0., 1., 2., 0., 0., 2., 2., 4., 2., 3., 3.,
        2., 4., 3., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 3., 2., 2., 1., 2.,
        0., 0., 3., 2.],
       [0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 3., 2., 0., 0., 0., 0.,
        1., 2., 0., 3.],
       [0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 1., 1., 1., 0., 0., 0.,
        0., 3., 0., 2.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1.,
        0., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 2., 0.,
        2., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        1., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
        0., 0., 3., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 1., 0.,
        1., 1., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
        0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1.,
        0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 2., 1.,
        2., 0., 2., 0.]], [[ 2.,  2.,  3.,  5.,  1.,  7.,  6., 13., 29., 27.,  9.,  6.,  6.,
         3.,  2.,  2.,  4.,  3.,  3.,  1.],
       [ 0.,  0.,  0.,  0.,  0.,  3.,  0.,  3., 10.,  7.,  9.,  5.,  3.,
         2.,  6.,  3.,  4.,  4.,  5.,  4.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  5.,  5.,  0.,  1.,  7.,
         2.,  0.,  3.,  3.,  1.,  4.,  7.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  2.,  0.,  1.,  2.,
         3.,  1.,  2.,  0.,  1.,  1.,  2.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  3.,  1.,
         0.,  0.,  1.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  3.,  1.,  0.,  2.,  0.,
         1.,  2.,  1.,  1.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  0.,  0.,
         1.,  0.,  0.,  1.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  1.,  0.,  1.,  1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         1.,  1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  1.,  0.,  0.,  0.,  0.,  0.]]]

def max_value(input_list):

    max_values = []
    for x in range(len(input_list)):
        current_category = input_list[x]
        max_values.append( max(current_category))
        pass
    max_value = max(max_values)
    return(max_value)

"""
Heatmap for each category is divided by the largest heatmap value of the category,
which produces a 2d space of values between 0 and 1, which will be our weights for categorization
"""

for category in all_heatmaps:

    max_val_of_category = max_value(category)
    #print("max_val_of_category = ", max_val_of_category)

    for row_list in category:

        for x in range(len(row_list)):
            current_val = row_list[x]  # find current value
            row_list[x] = current_val / max_val_of_category  # replace with value between 0 and 1

            pass

        pass

#print("All heatmap values = ", all_heatmaps)


