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




all_heatmaps = [[[ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  1.,  0.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  2.,  0.,  1.,  3.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  2.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  1.,  4.,  0.,  3.,  1.,  1.,  1.,  0.,  0.,  1.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  1.,  2.,  2.,  2.,  4.,  1.,  0.,  4.,  0.,  1.,  3.,
         0.,  0.,  0.,  0.,  0.,  0.,  1.],
       [ 0.,  4.,  7., 11.,  7.,  8.,  6.,  3.,  1.,  4.,  1.,  1.,  0.,
         1.,  1.,  0.,  0.,  0.,  1.,  0.],
       [ 0.,  3.,  2.,  9., 13.,  9.,  5.,  3., 14.,  2.,  1.,  2.,  1.,
         2.,  2.,  2.,  0.,  0.,  0.,  0.],
       [ 2.,  4.,  2.,  8., 11.,  6., 12.,  1.,  2.,  0.,  0.,  2.,  1.,
         0.,  1.,  1.,  0.,  0.,  0.,  0.],
       [ 1.,  7.,  3.,  5.,  3.,  6.,  7.,  1.,  0.,  1.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  1.,  4.,  4.,  1.,  3.,  0.,  1.,  0.,  1.,  0.,  0.,
         0.,  1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.]], [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 2.,  2.,  0.,  0.,  0.,  2.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  1.,  3.,  2.,  3.,  3.,  1.,  2.,  2.,  2.,  0.,  0.,  0.,
         0.,  1.,  1.,  0.,  0.,  0.,  1.],
       [ 4.,  0.,  3.,  2.,  7.,  5., 10.,  4.,  4.,  0.,  2.,  1.,  1.,
         0.,  1.,  1.,  2.,  0.,  0.,  1.],
       [ 1.,  2.,  5.,  6.,  6.,  7.,  3.,  2.,  7.,  3.,  4.,  6.,  3.,
         3.,  2.,  1.,  3.,  2.,  0.,  1.],
       [ 3.,  3.,  2.,  0.,  2.,  2.,  8.,  7.,  6.,  4.,  8.,  3.,  3.,
         1.,  0.,  1.,  1.,  1.,  1.,  1.],
       [ 3.,  2.,  3.,  4.,  5.,  1.,  4.,  5.,  2.,  8.,  4.,  1.,  1.,
         1.,  1.,  0.,  1.,  0.,  0.,  0.],
       [ 1.,  0.,  1.,  3.,  1.,  1.,  0.,  2.,  3.,  2.,  1.,  1.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  3.,  2.,  1.,  1.,  3.,  1.,  1.,  0.,  2.,  0.,
         1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  2.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,
         0.,  0.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.]], [[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 4.,  2.,  2.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [11.,  2.,  5.,  4.,  2.,  6.,  3.,  1.,  1.,  2.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 4., 12.,  6.,  6.,  7.,  5.,  3.,  2.,  1.,  0.,  2.,  4.,  2.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 2., 10.,  8.,  4.,  4.,  4.,  7.,  3.,  5.,  1.,  1.,  0.,  0.,
         0.,  0.,  0.,  0.,  1.,  1.,  1.],
       [ 2.,  4.,  7.,  4.,  5.,  2.,  3.,  3.,  2.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 2.,  4.,  6.,  5.,  7.,  5.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 2.,  6.,  4.,  0.,  6.,  1.,  2.,  0.,  1.,  0.,  0.,  1.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  3.,  3.,  2.,  2.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  3.,  1.,  1.,  1.,  0.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  3.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  3.,  3.,  2.,  2.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  3.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.]], [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [0., 2., 1., 0., 0., 2., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [1., 0., 1., 0., 1., 1., 2., 0., 1., 0., 1., 0., 1., 0., 0., 0.,
        0., 0., 0., 1.],
       [5., 3., 5., 2., 0., 1., 1., 1., 4., 1., 1., 0., 0., 0., 0., 0.,
        1., 0., 0., 0.],
       [2., 7., 4., 2., 4., 0., 2., 2., 5., 3., 1., 0., 2., 0., 1., 1.,
        1., 0., 0., 1.],
       [1., 2., 2., 0., 1., 2., 2., 2., 6., 6., 2., 6., 2., 0., 1., 1.,
        2., 0., 0., 1.],
       [1., 3., 6., 4., 6., 2., 2., 2., 6., 3., 3., 3., 1., 1., 0., 0.,
        0., 0., 0., 1.],
       [1., 3., 4., 4., 1., 4., 2., 4., 0., 3., 1., 1., 2., 0., 1., 0.,
        0., 1., 0., 1.],
       [3., 2., 1., 1., 4., 4., 1., 1., 6., 1., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [1., 1., 2., 2., 2., 4., 2., 2., 1., 2., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0.],
       [2., 1., 1., 3., 7., 3., 2., 2., 1., 2., 0., 0., 0., 0., 0., 0.,
        1., 0., 0., 0.],
       [0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [0., 1., 2., 3., 0., 0., 2., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [1., 0., 0., 1., 1., 0., 3., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [1., 0., 1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.]], [[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [0., 2., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0.],
       [0., 1., 2., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0.],
       [1., 1., 1., 0., 0., 2., 2., 0., 1., 1., 0., 1., 0., 0., 2., 0.,
        0., 0., 0., 0.],
       [1., 4., 3., 0., 0., 1., 2., 2., 0., 2., 1., 0., 2., 0., 1., 0.,
        0., 0., 0., 0.],
       [5., 6., 1., 1., 1., 3., 1., 0., 2., 0., 2., 1., 1., 0., 0., 0.,
        0., 0., 0., 1.],
       [3., 6., 1., 2., 1., 4., 2., 4., 0., 1., 2., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [3., 2., 4., 6., 6., 5., 7., 1., 2., 1., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [3., 5., 1., 9., 6., 3., 1., 0., 2., 0., 1., 1., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [2., 5., 4., 7., 5., 4., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [3., 2., 4., 3., 5., 4., 3., 2., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [3., 1., 0., 6., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [1., 0., 6., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [2., 2., 3., 4., 3., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [2., 2., 2., 2., 0., 3., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [0., 1., 4., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [1., 2., 6., 2., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.],
       [3., 5., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.]]]



all_xedges = [[-4.80380214, -4.29741186, -3.79102158, -3.2846313 , -2.77824102,
       -2.27185074, -1.76546046, -1.25907019, -0.75267991, -0.24628963,
        0.26010065,  0.76649093,  1.27288121,  1.77927148,  2.28566176,
        2.79205204,  3.29844232,  3.8048326 ,  4.31122288,  4.81761315,
        5.32400343],[-2.72902753, -2.24497233, -1.76091713, -1.27686193, -0.79280673,
       -0.30875153,  0.17530367,  0.65935887,  1.14341407,  1.62746927,
        2.11152447,  2.59557967,  3.07963487,  3.56369007,  4.04774527,
        4.53180047,  5.01585567,  5.49991087,  5.98396607,  6.46802127,
        6.95207647], [-3.82801604, -3.30457808, -2.78114013, -2.25770217, -1.73426421,
       -1.21082625, -0.6873883 , -0.16395034,  0.35948762,  0.88292557,
        1.40636353,  1.92980149,  2.45323945,  2.9766774 ,  3.50011536,
        4.02355332,  4.54699128,  5.07042923,  5.59386719,  6.11730515,
        6.6407431],[-2.05012774, -1.63787797, -1.2256282 , -0.81337843, -0.40112866,
        0.01112111,  0.42337088,  0.83562065,  1.24787042,  1.66012019,
        2.07236997,  2.48461974,  2.89686951,  3.30911928,  3.72136905,
        4.13361882,  4.54586859,  4.95811836,  5.37036813,  5.7826179 ,
        6.19486767], [-1.32954412, -0.91481879, -0.50009346, -0.08536813,  0.3293572 ,
        0.74408253,  1.15880785,  1.57353318,  1.98825851,  2.40298384,
        2.81770917,  3.2324345 ,  3.64715983,  4.06188516,  4.47661048,
        4.89133581,  5.30606114,  5.72078647,  6.1355118 ,  6.55023713,
        6.96496246]]

all_yedges = [[ 468.42041998,  776.19217826, 1083.96393654, 1391.73569482,
       1699.5074531 , 2007.27921138, 2315.05096966, 2622.82272794,
       2930.59448622, 3238.3662445 , 3546.13800278, 3853.90976106,
       4161.68151934, 4469.45327762, 4777.2250359 , 5084.99679418,
       5392.76855246, 5700.54031074, 6008.31206902, 6316.0838273 ,
       6623.85558558],[ 127.80601303,  389.9871669 ,  652.16832077,  914.34947463,
       1176.5306285 , 1438.71178237, 1700.89293624, 1963.07409011,
       2225.25524398, 2487.43639785, 2749.61755172, 3011.79870558,
       3273.97985945, 3536.16101332, 3798.34216719, 4060.52332106,
       4322.70447493, 4584.8856288 , 4847.06678267, 5109.24793653,
       5371.4290904], [ 241.44876847,  675.16123241, 1108.87369634, 1542.58616027,
       1976.2986242 , 2410.01108813, 2843.72355207, 3277.436016  ,
       3711.14847993, 4144.86094386, 4578.57340779, 5012.28587173,
       5445.99833566, 5879.71079959, 6313.42326352, 6747.13572746,
       7180.84819139, 7614.56065532, 8048.27311925, 8481.98558318,
       8915.69804712], [356.03677902,  586.97568294,  817.91458686, 1048.85349078,
       1279.7923947 , 1510.73129862, 1741.67020254, 1972.60910646,
       2203.54801038, 2434.4869143 , 2665.42581822, 2896.36472214,
       3127.30362606, 3358.24252998, 3589.18143391, 3820.12033783,
       4051.05924175, 4281.99814567, 4512.93704959, 4743.87595351,
       4974.81485743], [190.30060854,  623.57439602, 1056.8481835 , 1490.12197098,
       1923.39575846, 2356.66954594, 2789.94333342, 3223.2171209 ,
       3656.49090838, 4089.76469586, 4523.03848334, 4956.31227082,
       5389.5860583 , 5822.85984578, 6256.13363326, 6689.40742074,
       7122.68120822, 7555.9549957 , 7989.22878318, 8422.50257066,
       8855.77635814]]




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


#print(all_heatmaps)
