#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from scipy.signal import find_peaks
import os
import glob
import json
import pandas as pd
 

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
            db_ind = binarySearch_tol(np.append(peaks,np.max(peaks)+5),
                                      0, len(peaks)-1, exp_peak,tolerance)
            
            
            # TODO add an option to consider all edges, not only the max
            if db_ind != -1:
                if mass_diff[j] >0:
                    selected_index = db_ind[np.argmax(intensity[db_ind])]
                    spectrum_edge_param.append([i,selected_index,j])
                
                if mass_diff[j] <=0:
                    selected_index = db_ind[np.argmax(intensity[db_ind])]
                    spectrum_edge_param.append([selected_index,i,j])
    
    return(spectrum_edge_param)

def profile_to_cent(mzs,intensities,mass_range,max_peaks,centroid):
        
        intensity_arr = np.array(intensities)
        index_mass = (mzs >= mass_range[0]) & (mzs <= mass_range[1])
        mzs = mzs[index_mass]
        intensity_arr = intensity_arr[index_mass]

        # Peak detection
        if centroid == False:
            peaks, _ = find_peaks(intensity_arr,distance=3,width=2)
        else:
            peaks = np.arange(0,len(intensity_arr),1)
            
        
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
my_parser.add_argument('-i','--input', action='store', type=str, required=True,help='json file')


args = my_parser.parse_args()

param_json_path = args.input

# load parameters
param = []
with open(param_json_path) as json_file:
    param = json.load(json_file)


output_dir = param['output_dir']
msi_dir = param['msi_dir']
tolerance = param['tolerance']
mass_diff = param['mass_diff']
max_peaks = param['max_peaks']
annot_table_name = param['annot_table']
mass_range = param["mass range"]
min_mass = mass_range[0]
max_mass = mass_range[1]


annot_table = pd.read_csv(annot_table_name)  

# create a directory for the parameters
os.mkdir(output_dir + os.path.splitext(os.path.basename(param_json_path))[0])
output_dir= output_dir + os.path.splitext(os.path.basename(param_json_path))[0] + "/"

print("Directory '% s' created" % output_dir)


if param['file_type'] == 'imzML':
    
    for it in os.scandir(msi_dir):
        if it.is_dir():
            msi_class_dir = it.path
            msi_class = os.path.basename(it.path)
            if np.sum(annot_table["MSI name"] == msi_class) >0:
                sub_df = annot_table.loc[annot_table["MSI name"] == msi_class].copy()
                #sub_df = sub_df.sort_values('origianl MSI pixel id').copy()

                os.mkdir(output_dir + msi_class)

                # Iterate through the imzml
                ct_spectrum = 0
                iterator_df = sub_df["origianl MSI pixel id"].to_numpy()[0]
                massspec_id = 0
                #sub_df["origianl MSI pixel id"].to_numpy()[0]

                for file in glob.glob(msi_class_dir + "/*.imzML"):

                    p = ImzMLParser(file)
                    for idx, (x,y,z) in enumerate(p.coordinates):
                        print(idx,(x,y,z),np.shape(sub_df)[0],iterator_df)

                        if ct_spectrum ==iterator_df:

                            mzs, intensities = p.getspectrum(idx)
                            spectrum = profile_to_cent(mzs,intensities,mass_range,max_peaks,True)
                            np.save(output_dir+ msi_class +"/" + 'spec_' + str(massspec_id), spectrum)

                            spectrum_edge_param = generate_graph_from_spectrum(spectrum,mass_diff,tolerance)
                            np.save(output_dir+ msi_class +"/" + 'graph_' + str(massspec_id), spectrum_edge_param)

                            massspec_id += 1
                            # todo change and put in the first condition 
                            if massspec_id< np.shape(sub_df)[0]:
                                iterator_df = sub_df["origianl MSI pixel id"].to_numpy()[massspec_id]
                        ct_spectrum += 1                    
        
else:
    

    # Iterate through a diretory of csv files 
    for it in os.scandir(msi_dir):
        if it.is_dir():
            msi_class_dir = it.path
            msi_class = os.path.basename(it.path)

            os.mkdir(output_dir + msi_class)

            massspec_id = 0
            for file in glob.glob(msi_class_dir + "/*.csv"):
                spectrum = np.genfromtxt(file, delimiter=',')

                mzs = spectrum[:,0]
                intensities = spectrum[:,1]

                spectrum = profile_to_cent(mzs,intensities,max_peaks,False)
                np.save(output_dir+ msi_class +"/" + 'spec_' + str(massspec_id), spectrum)

                spectrum_edge_param = generate_graph_from_spectrum(spectrum,mass_diff,tolerance)
                np.save(output_dir + msi_class +"/"+ 'graph_' + str(massspec_id), spectrum_edge_param)
                print(massspec_id)
                massspec_id += 1 