#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from numpy import genfromtxt
from pyimzml.ImzMLParser import ImzMLParser
from scipy.stats import norm

def create_mean_spectrum(msi_file,min_mz,max_mz,precision):
    # gaussian distribution
    
    x_values = np.arange(-precision, precision, 1)
    
    mean = 0
    standard_deviation = precision/4
    y_values = norm(mean, standard_deviation)
    p = ImzMLParser(msi_file)
   
    
    mean_spectrum_mz = np.arange(min_mz, max_mz, 0.0001)
    mean_spectrum_int = np.zeros(np.size(mean_spectrum_mz,0))
    
    i = 0 
    for idx, (x,y,z) in enumerate(p.coordinates):
        m, intensities = p.getspectrum(idx)
        
        if len(intensities) >2:
            mzs = m[np.logical_and(m>min_mz,m<max_mz)]
            intensities = intensities[np.logical_and(m>min_mz,m<max_mz)]



            intensities_norm = intensities/np.sum(intensities)
            intensities_norm[intensities_norm <0] = 0
            mzs_ind = np.rint(mzs*10000).astype('int')


            for s in x_values:
                mean_spectrum_int[mzs_ind+s] += intensities_norm*y_values.pdf(s)
                #mean_spectrum_int[mzs_ind+s] += intensities_norm

        
        
    mzs = np.where(mean_spectrum_int >0)[0]/10000
    intensities = mean_spectrum_int[np.where(mean_spectrum_int >0)[0]]
    #mzs = mean_spectrum_mz
    #intensities = mean_spectrum_int
    return(mzs,intensities)


my_parser = argparse.ArgumentParser(allow_abbrev=False)
my_parser.add_argument('-i','--input', action='store', type=str, required=True,help='imzML file')
my_parser.add_argument('-p','--precision', action='store', type=int, required=True,help='precision value in Da*10000')
my_parser.add_argument('-o1','--output1', action='store', type=str, required=True,help='ouput folder')


args = my_parser.parse_args()

msi_file = args.input
precision = args.precision
output_file = args.output1

min_mz = 0
max_mz = 20000
# precision = 100 -> 100/10000 Da
mzs, ints = create_mean_spectrum(msi_file ,min_mz,max_mz,precision)
t =  np.vstack((mzs, ints))
np.save(output_file, t)

    