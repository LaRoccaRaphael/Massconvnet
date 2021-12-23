#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from scipy.signal import find_peaks
import os

#python function to convert profile imzML to multiple files containing single centroided spectrum
def generate_kendrick_param(peaks):
    md = 1
    #md = 14/14.0156500641
    km = peaks*md
    kmd = km - np.ceil(km)
    
    return(km,kmd)

def compute_masserror(experimental_mass, database_mass, tolerance):
    # mass error in Dalton
    if database_mass != 0:
        return abs(experimental_mass - database_mass) <= tolerance

def binarySearch_tol(arr, l, r, x, tolerance): 
    # binary search with a tolerance (mass) search from an ordered list of masses
    while l <= r: 
        mid = l + (r - l)//2; 
        if compute_masserror(x,arr[mid],tolerance): 
            itpos = mid +1
            itneg = mid -1
            index = []
            index.append(mid)
            if( itpos < len(arr)):
                while compute_masserror(x,arr[itpos],tolerance) and itpos < len(arr):
                    index.append(itpos)
                    itpos += 1 
            if( itneg > 0): 
                while compute_masserror(x,arr[itneg],tolerance) and itneg > 0:
                    index.append(itneg)
                    itneg -= 1     
            return index 
        elif arr[mid] < x: 
            l = mid + 1
        else: 
            r = mid - 1
    return -1

def generate_graph_from_spectrum(spectrum,mass_diff,tolerance): 
    
    peaks = spectrum[:,0]
    intensity = spectrum[:,1]
    
    spectrum_edge_param = list()
    
    # compute graph parameters edge index pairs and edge type
    for i in range(0,np.size(peaks,0)):
        for j in range(0,np.size(mass_diff,0)):
            
            exp_peak = peaks[i]+mass_diff[j]
            
            # peaks need to be ordered
            db_ind = binarySearch_tol(np.append(peaks,np.max(peaks)+1),
                                      0, len(peaks)-1, exp_peak,tolerance)
            
            if db_ind != -1:
                selected_index = db_ind[np.argmax(intensity[db_ind])]
                spectrum_edge_param.append([i,selected_index,j])
    

    return(spectrum_edge_param)

def profile_to_cent(massspec_id,mzs,intensities,max_peaks):
        intensity_arr = np.array(intensities)

        # Peak detection
        peaks, _ = find_peaks(intensity_arr,distance=10,width=5)
        #print(len(peaks))

        # Select the max_peaks most intense peaks
        peaks = peaks[np.argsort(intensity_arr[peaks])[::-1][:max_peaks]]
        selected_mzs = mzs[peaks]
        selected_intensity_arr = intensity_arr[peaks]

        # Sort according to the mass 
        new_mzs = selected_mzs[np.argsort(selected_mzs)]
        new_intensity_arr = selected_intensity_arr[np.argsort(selected_mzs)]

        # Compute kendrick params
        km,kmd = generate_kendrick_param(new_mzs)
        new_spectrum = np.vstack((km,kmd, new_intensity_arr)).T    
        
        return(new_spectrum)
    

# Load the arguments
my_parser = argparse.ArgumentParser(allow_abbrev=False)
my_parser.add_argument('-i','--input', action='store', type=str, required=True,help='msi input file')
my_parser.add_argument('-mp','--max_peaks', action='store', type=int, required=True,help='maximum number of peaks allow per spectrum')
my_parser.add_argument('-o','--output', action='store', type=str, required=True,help='ouput dir')

args = my_parser.parse_args()

msi = args.input
output_dir = args.output
max_peaks = args.max_peaks

mass_diff = [28.0313,14.01565,2.01565,15.99491464,21.981945,37.955885,189.0426,1.0033,2.0067,26.01565007,43.98982928,44.99765432,18.01056471,27.99491464,16.01872408,15.01089905,13.97926]
tolerance = 0.001

# Write the new ouput directory
filename = os.path.basename(msi).split('.imzML')[0]
new_dir = output_dir + filename + "_cent_p_" + str(max_peaks) +"/"
os.mkdir(new_dir)
print("Directory '% s' created" % new_dir)

# Iterate through the imzml
p = ImzMLParser(msi)
massspec_id = 0
for idx, (x,y,z) in enumerate(p.coordinates):
    print(massspec_id)
        
    mzs, intensities = p.getspectrum(idx)
    spectrum = profile_to_cent(massspec_id,mzs,intensities,max_peaks)
    np.save(new_dir + 'spec_' + str(massspec_id), spectrum)

    spectrum_edge_param = generate_graph_from_spectrum(spectrum,mass_diff,tolerance)
    np.save(new_dir + 'graph_' + str(massspec_id), spectrum_edge_param)
    
    massspec_id += 1 
