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




def find_rhythmic_packets(input_similar_magnitude_locations, input_similar_magnitude_values, nearest_peak_cutoff=200):

    """
    find "packets" of peaks that are close together in time
    these should be in a "similar" peak list
    they should be grouped near each other
    "rhythm rate" of the whole file could be the shorted peak-to-peak distance?

    tolerance for packet distribution?
    return average peak distance within packet?

    Imagine the output list as such:

    [[[1, 2], [3]], [[4]], [[5, 6, 7, 8], [9, 10, 11, 12]]]
    [val] = list 3 (smallest unit)
    [[val]] = list 2
    [[[val]]] = list 1  (largest list unit)
    
    """
    # create destructible input lists that won't affect the original input lists
    used_input_similar_magnitude_locations = []
    used_input_similar_magnitude_values = []
    for loc, val in zip(input_similar_magnitude_locations, input_similar_magnitude_values) :
        used_input_similar_magnitude_locations.append(loc)
        used_input_similar_magnitude_values.append(val)
        pass

        
    # to find a "packet" of nearby peak points:
    # read the input similar magnitude peak locations
    # within a similar peak location list:
    # take the first point, and see if the next value along is within the nearest_peak_cutoff limit
    # if it is, add it to a packet sublist, remove the other values from the larger list
    # if it isn't, create a new packet sublist, add the next value to it

    #print("Rhythmic Packet detection starting")
    #print("Input peak locations = ", used_input_similar_magnitude_locations)
    #print("Input peak magnitudes = ", used_input_similar_magnitude_values)
    #print("Nearest peak within packet cutoff value = ", nearest_peak_cutoff)

    all_packet_locations = []  # largst lists containing all packet arrays in the file
    all_packet_values = []  # we'll call this "List 1", it's the highest order list

    packet_locations = [(used_input_similar_magnitude_locations[0])[0]]  # update first peak into loc and value arrays
    packet_values = [(used_input_similar_magnitude_values[0])[0]]  # this is the smallest order list, let's call it "List 3"
        
    for n in range(len(used_input_similar_magnitude_locations)):
        similar_magnitude_location_packets = []  # this is the middle tier list, let's call it "List 2"
        similar_magnitude_value_packets = []
        
        packet_locations = []
        packet_values = []
        
        for x in range(len(used_input_similar_magnitude_locations[n])):
            #print("Sublist index in larger array = ", n)
            #print("Peak location index in sublist = ", x)
            current_peak_loc = (used_input_similar_magnitude_locations[n])[x] # current locations and peak magnitude
            current_peak_val = (used_input_similar_magnitude_values[n])[x]

            if len(used_input_similar_magnitude_locations[n]) > 1:
                # ensures list contains more than 1 value (otherwise it's a packet of 1 on it's own)
                
                if current_peak_loc == (used_input_similar_magnitude_locations[n])[-1]:
                    # if x is last value, it has already been sorted by the x-1 value
                    similar_magnitude_location_packets.append(packet_locations)
                    similar_magnitude_value_packets.append(packet_values)
                    pass
                
                else:
                    if x == 0:
                        # if first value in the sublist
                        packet_locations = [(used_input_similar_magnitude_locations[n])[x]]  # update first peak into loc and value arrays
                        packet_values = [(used_input_similar_magnitude_values[n])[x]]
                        pass
                    else:
                        pass
                    next_peak_loc = (used_input_similar_magnitude_locations[n])[x+1]  # next peak location and magnitude along in the larger list
                    next_peak_val = (used_input_similar_magnitude_values[n])[x+1]

                    if next_peak_loc <= (round(float(current_peak_loc) + float(nearest_peak_cutoff), 0)):
                        # if value is near to previous value
                        #print("Value within cutoff, appended to existing packet")
                        packet_locations.append(next_peak_loc)  # put this next value into the packet sublist
                        packet_values.append(next_peak_val)
                        pass

                    elif next_peak_loc > (round(float(current_peak_loc) + float(nearest_peak_cutoff), 0)):
                        #print("Value not within cutoff, new packet created")
                        # if value is not near to previous value, make a new packet within this list of similar peaks
                        similar_magnitude_location_packets.append(packet_locations)
                        similar_magnitude_value_packets.append(packet_values)
                        packet_locations = [next_peak_loc]  # sublists containing individual packet arrays
                        packet_values = [next_peak_val]            
                        pass
                    pass
                
                pass
            
            elif len(used_input_similar_magnitude_locations[n]) <= 1:
                packet_locations = [(used_input_similar_magnitude_locations[n])[x]]  # update first peak into loc and value arrays
                packet_values = [(used_input_similar_magnitude_values[n])[x]]  # list 3
                similar_magnitude_location_packets.append(packet_locations)  # list 2
                similar_magnitude_value_packets.append(packet_values)        
                pass
            else:
                pass
                
            pass
        all_packet_locations.append(similar_magnitude_location_packets) # list 1
        all_packet_values.append(similar_magnitude_value_packets)
        packet_locations = []
        packet_values = []

    #print("Packet locations = ", all_packet_locations)
    #print("Packet values = ", all_packet_values)

    return(all_packet_locations, all_packet_values)

    pass
