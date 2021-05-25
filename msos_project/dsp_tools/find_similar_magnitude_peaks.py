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

"""
WHAT ABOUT in the case that there are weird peak values inbetween your current similarity list
and the next similar peak??

- split the file into similarity bands initially?
- instead do a final sweep of this style which collates similar magnitude bands?

"""

def find_similar_magnitude_peaks(input_peak_locations, input_peak_values, magnitude_similarity=0.9):
    """
    used to find single peak values which are similar in amplitude
    """
    similar_magnitude_locations = []  # list which will contain sublists of similar magnitude peak locations!
    similar_magnitude_values = []  # list which will contain sublists of similar magnitude peak values!

    similar_magnitude_locations_sublist = []
    similar_magnitude_values_sublist = []  # these lists will store one set of similar values
    # if other values are quite different they will be put in a separate list

    assigned_peak_locations = input_peak_locations[:]
    assigned_peak_values = input_peak_values[:]

    x = 0

    while len(assigned_peak_locations) > 0:
        #print("Length of assigned peak locations = ", len(assigned_peak_locations))
        #print("")
        #print("")

        similar_magnitude_locations_sublist = []  # initialise sublist for locations
        similar_magnitude_values_sublist = []  # initialise sublist for values
        locations_to_be_removed = []
        values_to_be_removed = []

        for n in range(len(assigned_peak_locations)):
            #print("N = ", n)
            #print("assigned peak locations length =", len(assigned_peak_locations))
            peak_location = assigned_peak_locations[n]  # find value and location of a peak
            peak_value = assigned_peak_values[n]
            #print("peak location = ", peak_location)
            
            if n == 0:
                similar_magnitude_locations_sublist.append(peak_location)  # append value to similar magnitude sublist
                similar_magnitude_values_sublist.append(peak_value)
                locations_to_be_removed.append(peak_location)  # this is used to "remember" to remove the first peak value from the input list later
                values_to_be_removed.append(peak_value)
                pass
            else:
                current_sublist_average = (sum(similar_magnitude_values_sublist)/ len(similar_magnitude_values_sublist)) # find average of current sublist
                diff = abs(float(current_sublist_average) - float(peak_value))  # difference between current sublist average and next input peak value
                #print("Difference between sublist average and current val = ", diff)

                tolerance_val = float(1) - float(magnitude_similarity)
                upper_lim_ratio = 1 + tolerance_val  # this is the allowed upper limit ratio of the list average to the next value
                lower_lim_ratio = 1 - tolerance_val
                
                upper_lim = (upper_lim_ratio*current_sublist_average)  # values which decide whether next value will be admitted to similar locations sublist
                lower_lim = (lower_lim_ratio*current_sublist_average)

                #print("Peak value = ", peak_value)
                #print("upper limit = ", upper_lim)
                #print("lower limit = ", lower_lim)
                
                if peak_value >= lower_lim and peak_value <= upper_lim:
                    # if value is similar to the tolerance magnitude, we put it in a sublist of "similar" peak values
                    similar_magnitude_locations_sublist.append(peak_location)  # append value to similar magnitude sublist
                    similar_magnitude_values_sublist.append(peak_value)
                    
                    locations_to_be_removed.append(peak_location)  # remember which values need to be taken out of the input list
                    values_to_be_removed.append(peak_value)
                    #print("Value removed")

                    pass

                elif peak_value < lower_lim or peak_value > upper_lim:
                    # ignore the value if it is not similar to the current "similar" list, it will be picked up in a later pass
                    #print("Value passed")
                    pass

                else:
                    print("Error in detection peak value, or tolerance limits in find_similar_magnitude_peaks")
                    pass

                #print("Sublist locations = ", similar_magnitude_locations_sublist)
                pass
            pass

        
        for removed_val in range(len(locations_to_be_removed)):
            assigned_peak_locations.remove(locations_to_be_removed[removed_val])  # remove value(location) from input list!
            assigned_peak_values.remove(values_to_be_removed[removed_val])  # remove peak value from input list
            pass
            
        similar_magnitude_locations.append(similar_magnitude_locations_sublist) # list which will contain sublists of similar magnitude peak locations!
        similar_magnitude_values.append(similar_magnitude_values_sublist)  # list which will contain sublists of similar magnitude peak values!
        pass

    
    #print("Similar magnitude peaks (locations) = ", similar_magnitude_locations)
    #print("Similar magnitude peaks (magnitudes) = ", similar_magnitude_values)

    #print("Input peak locations = ", input_peak_locations)

    return(similar_magnitude_locations, similar_magnitude_values)
    
    pass


"""
### depreceated older version of this function!

def find_similar_magnitude_peaks212e2(input_peak_locations, input_peak_values, magnitude_similarity=0.9):
    similar_magnitude_locations = []  # list which will contain sublists of similar magnitude peak locations!
    similar_magnitude_values = []  # list which will contain sublists of similar magnitude peak values!

    similar_magnitude_locations_sublist = []
    similar_magnitude_values_sublist = []  # these lists will store one set of similar values
    # if other values are quite different they will be put in a separate list
    
    for n in range(len(input_peak_locations)):
        print("n = ", n)
        # only for the first value (n=0)
        peak_value = input_peak_values[n]  # find value and location of a peak
        peak_location = input_peak_locations[n]
        
        if n == 0:
            pass

        elif n != 0:
            # for any other value
            current_sublist_average = (sum(similar_magnitude_values_sublist)/ len(similar_magnitude_values_sublist)) # find average of current sublist

            diff = abs(float(current_sublist_average) - float(peak_value))  # difference between current sublist average and next input peak value
            print("Difference between sublist average and current val = ", diff)

            tolerance_val = float(1) - float(magnitude_similarity)
            upper_lim_ratio = 1 + tolerance_val  # this is the allowed upper limit ratio of the list average to the next value
            lower_lim_ratio = 1 - tolerance_val
            
            upper_lim = (upper_lim_ratio*current_sublist_average)  # values which decide whether next value will be admitted to similar locations sublist
            lower_lim = (lower_lim_ratio*current_sublist_average)

            print("tolerance value applied = ", tolerance_val)
            print("upper limit = ", upper_lim)
            print("lower limit = ", lower_lim)

            if diff <= upper_lim and diff >= lower_lim:
                pass

            elif diff > upper_lim or diff < lower_lim:
                similar_magnitude_locations.append(similar_magnitude_locations_sublist)  # store last sublist in the larger list
                similar_magnitude_values.append(similar_magnitude_values_sublist)
                similar_magnitude_locations_sublist = []  # create a new sublist for the next sample
                similar_magnitude_values_sublist = []
                pass

            else:
                print("Error in detecting diff or tolerance_val in find_similar_magnitude_peaks() in rhythmdetection()")
                pass

            pass

        else:
            print("Error in detecting n value in find_similar_magnitude_peaks() in rhythmdetection()")
            pass
        
        similar_magnitude_locations_sublist.append(peak_location)  # store first peak value and location in a sublist
        similar_magnitude_values_sublist.append(peak_value)
        
        pass
    
    similar_magnitude_locations.append(similar_magnitude_locations_sublist)  # store last sublist in the larger list
    similar_magnitude_values.append(similar_magnitude_values_sublist)

    # collect similar sublists here
        
            
    print("Similar magnitude peaks (locations) = ", similar_magnitude_locations)
    print("Similar magnitude peaks (magnitudes) = ", similar_magnitude_values)

    return(similar_magnitude_locations, similar_magnitude_values)
        
    pass


"""
