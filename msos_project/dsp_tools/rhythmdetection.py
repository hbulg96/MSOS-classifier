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
import msos_project.dsp_tools.find_similar_magnitude_peaks as find_similar_magnitude_peaks
import math
from math import sqrt
from numpy import std


def rhythmdetection(input_packet_locations,
                    input_packet_values,
                    plot_results=False):


    #print("rhythm detection Input packet locations = ", input_packet_locations)
    #print("rhythm detection Input packet values = ", input_packet_values)
    
    # first, how far apart are the peak_points?
    # are all of the peak_points a similar distance apart?
    # are peak values of a similar magnitude a similar distance apart?
    individual_packet_mean_time_intervals, individual_packet_standard_deviations = rhythmic_tendency_of_packets(input_packet_locations,
                                                                                                                input_packet_values)

    # low standard deviation could indicate they are reasonably averagely spaced
    # also smallest time difference in the packets could represent the smallest rhythmic unit!! 

    # second, how far apart are the packets themselves?
    # using the first value of each packet, what is the distance of the packets?
    # what is the standard deviation of distance between packets?
    # low standard deviation could indicate they are reasonably averagely spaced
    
    # return values that are indicative of rhythm for instance:
    # average time between peaks
    # how close is the mean difference to each difference (variation)
    # are there different sets of peak magnitudes?

    average_time_diff_between_packets, standard_deviation_between_packets = distance_between_packets_of_similar_amplitude(input_packet_locations,
                                                                                                                            input_packet_values)


    """
    rhythmic_packet_locations, rhythmic_packet_values = find_rhythmic_packets(similar_magnitude_locations,
                                                                              similar_magnitude_values,
                                                                              rhythmic_packet_cutoff)

    """

    within_packet_rhythm_tendency, between_packet_rhythm_tendency = overall_rhythm_tendency(individual_packet_mean_time_intervals,
                                                                                    individual_packet_standard_deviations,
                                                                                    average_time_diff_between_packets,
                                                                                    standard_deviation_between_packets)

    if plot_results == True:
        input_file = numpy.array(input_file[1], dtype = int)  # create numpy array of "original" input file
        matplotlib.pyplot.plot(input_file, label= "input_file")  # plot input waveform
        matplotlib.pyplot.plot(initial_peak_points,initial_peak_values, label= "peak_values")  # plot detected peak values
        pyplot.xlabel("Time")
        pyplot.ylabel("Amplitude")
        #pyplot.legend()

        for similar_peak_loc, similar_peak_val in zip(similar_magnitude_locations, similar_magnitude_values):
            matplotlib.pyplot.plot(similar_peak_loc, similar_peak_val, label= "similar_peak_values", marker="o")
            pass

            
        #matplotlib.pyplot.plot(flattened_peak_points, flattened_peak_values, label= "flat_peak_values", marker="x")  # plot flattened peak values
        pyplot.xlabel("Time")
        pyplot.ylabel("Amplitude")
        #pyplot.legend()
        pyplot.show()
        pass

    else:
        pass

    
    return(within_packet_rhythm_tendency, between_packet_rhythm_tendency)



def overall_rhythm_tendency(individual_packet_mti, individual_packet_sd, between_packet_mti, between_packet_sd):

    if len(individual_packet_sd) > 1:
        in_packet_smallest_standard_deviation = []
        
        for standard_deviation in individual_packet_sd:

            if standard_deviation == 0:
                #filter out 0's from standard deviations
                pass

            else:
                in_packet_smallest_standard_deviation.append(standard_deviation)
                pass
            pass

        if len(in_packet_smallest_standard_deviation) > 0:
            in_packet_smallest_standard_deviation = min(in_packet_smallest_standard_deviation) # smallest non-zero value
            # based on a y = mx+c line where 4000 is 0 tendency, and near 0 is 1 tendency

            within_packet_rhythm_tendency = ((-0.00025)*in_packet_smallest_standard_deviation) + 1
            if within_packet_rhythm_tendency < 0:
                within_packet_rhythm_tendency = 0
                pass
            else:
                pass
            pass

        else:
            within_packet_rhythm_tendency = 0
            pass



        between_packet_smallest_standard_deviation = []

        for standard_deviation in between_packet_sd:
            if standard_deviation == 0:
                #filter out 0's from standard deviations
                pass

            else:
                between_packet_smallest_standard_deviation.append(standard_deviation)
                pass
            pass
        if len(between_packet_smallest_standard_deviation) > 0:
            between_packet_smallest_standard_deviation = min(between_packet_smallest_standard_deviation)

            between_packet_rhythm_tendency = ((-0.00025)*between_packet_smallest_standard_deviation) + 1

            if between_packet_rhythm_tendency < 0:
                between_packet_rhythm_tendency = 0
                pass
            else:
                pass
            pass
        else:
            between_packet_rhythm_tendency = 0
            pass
        pass

    else:
        within_packet_rhythm_tendency = 0
        between_packet_rhythm_tendency = 0
        
    #print("within packet rhythm tendency = ", within_packet_rhythm_tendency)
    #print("between packet rhythm tendency = ", between_packet_rhythm_tendency)

    # NOTE AT CURRENT THESE ARE THE *BEST* VALUES OF TENDENCY BASED ON THE SMALLEST SD VALUE
    
    return(within_packet_rhythm_tendency, between_packet_rhythm_tendency)


def rhythmic_tendency_of_packets(input_packet_locations_list, input_packet_values_list):

    #print("Rhythmic tendency within packets")
    
    all_time_averages_within_packets = []
    all_standard_deviations_within_packets = []
    
    for similar_magnitude_locations, similar_magnitude_values in zip(input_packet_locations_list, input_packet_values_list):

        for packet_locations, packet_values in zip(similar_magnitude_locations, similar_magnitude_values):


            time_difference_between_peaks = []  # initialise time_difference list
            
            if len(packet_locations) > 1:
                # if the packet has more than one value in it
                
                for x in range(len(packet_locations)):
                    #print("Peak location within packet = ", packet_locations[x])
                    #print("Peak value within packet = ", packet_values[x])
                    if x == 0:
                        # ignore the first value, we're calculating difference in peak locations so we'll use the second value in the list to start
                        pass
                    else:
                        # find time interval between the two peaks
                        time_difference = packet_locations[x] - packet_locations[x-1]
                        time_difference_between_peaks.append(time_difference)
                        pass

                    pass

                #print("Time difference between peaks = ", time_difference_between_peaks)

                mean_time_difference_in_packet = sum(time_difference_between_peaks) / len(time_difference_between_peaks)
                # average time difference in packet
                
                """
                sum_of_peak_values_minus_mean_value = 0
                
                for time_interval in time_difference_between_peaks:
                    sum_of_peak_values_minus_mean_value += (time_interval - mean_time_difference_in_packet)**2
                    # from standard deviation calculation, sum squared differences of each value to mean value
                    pass
                    
                standard_deviation_of_time_intervals = math.sqrt((sum_of_peak_values_minus_mean_value)/ len(time_difference_between_peaks))
                # from standard deviation calculation, divide sum of squared differences by population size, then square root
                """
                # hand calculation of standard deviation. replaced with numpy std. as more repeatable
                
                standard_deviation_of_time_intervals = numpy.std(time_difference_between_peaks)
                
                #print("Mean time difference in packet = ", mean_time_difference_in_packet)
                #print("Standard deviation of time intervals = ", standard_deviation_of_time_intervals)
                
                all_time_averages_within_packets.append(mean_time_difference_in_packet)
                all_standard_deviations_within_packets.append(standard_deviation_of_time_intervals)

                pass
            else:
                pass
            pass

        #print("All time difference averages within packets = ", all_time_averages_within_packets)
        #print("All standard deviations within packets = ", all_standard_deviations_within_packets)
        return(all_time_averages_within_packets, all_standard_deviations_within_packets)
    

    # ensure the locations are in order

    # what is the average distance between each peak in the packet?

    # what is the standard deviation in distances? (low SD means more likely to be rhythmic)

    pass


def distance_between_packets_of_similar_amplitude(input_packet_locations_list, input_packet_values_list):

    #print("Distance between packets of similar amplitude")
    
    average_distances_between_packets_of_sim_magnitude = []
    standard_deviations_of_distance_between_packets_of_sim_magnitude = []
    
    for similar_magnitude_locations, similar_magnitude_values in zip(input_packet_locations_list, input_packet_values_list):
        
        #print("Similar magnitude packets = ", similar_magnitude_locations)

        first_values_in_each_packet = []
        
        for x in range(len(similar_magnitude_locations)):
            packet_locations = similar_magnitude_locations[x]

            #print("packet locations = ", packet_locations)

            first_values_in_each_packet.append(packet_locations[0])
            pass

        difference_between_first_peaks_in_packet = []

        if len(first_values_in_each_packet) > 1:
            # ensures list has more than one value in it!
            for n in range(len(first_values_in_each_packet)):

                if n == 0:
                    # ignore first value, we are looking for difference in values!
                    pass

                else:
                    difference_val = (first_values_in_each_packet[n] - first_values_in_each_packet[n-1])
                    difference_between_first_peaks_in_packet.append(difference_val)
                    #print("Difference in first peak values = ", difference_val)
                    pass
                pass


            average_distance_between_packets = sum(difference_between_first_peaks_in_packet)/len(difference_between_first_peaks_in_packet)

            standard_deviation_of_packet_intervals = numpy.std(difference_between_first_peaks_in_packet)

            average_distances_between_packets_of_sim_magnitude.append(average_distance_between_packets)
            standard_deviations_of_distance_between_packets_of_sim_magnitude.append(standard_deviation_of_packet_intervals)
            pass

        else:
            pass

        pass

    #print("Average distances between rhythmic packet first values = ", average_distances_between_packets_of_sim_magnitude)
    #print("Standard deviation in distances = ", standard_deviations_of_distance_between_packets_of_sim_magnitude)

    
    return(average_distances_between_packets_of_sim_magnitude, standard_deviations_of_distance_between_packets_of_sim_magnitude)

    
"""
def find_rhythmic_packets(input_similar_magnitude_locations, input_similar_magnitude_values, nearest_peak_cutoff=200):

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

    print("Rhythmic Packet detection starting")
    print("Input peak locations = ", used_input_similar_magnitude_locations)
    print("Input peak magnitudes = ", used_input_similar_magnitude_values)
    print("Nearest peak within packet cutoff value = ", nearest_peak_cutoff)

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

    print("Packet locations = ", all_packet_locations)
    print("Packet values = ", all_packet_values)

    return(all_packet_locations, all_packet_values)

    pass
"""




