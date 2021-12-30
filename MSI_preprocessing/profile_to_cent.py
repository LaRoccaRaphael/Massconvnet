#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from scipy.signal import find_peaks

#python function to convert profile imzML to centroided imzML


def profile_to_cent(msi,output_file):
    with ImzMLWriter(output_file) as w:
        p = ImzMLParser(msi)
        for idx, (x,y,z) in enumerate(p.coordinates):
            mzs, intensities = p.getspectrum(idx)
            intensity_arr = np.array(intensities)
            peaks, _ = find_peaks(intensity_arr,distance=3,width=2)
            peaks_mz = mzs[peaks]
            w.addSpectrum(mzs[peaks], intensity_arr[peaks], (x,y,z))


my_parser = argparse.ArgumentParser(allow_abbrev=False)
my_parser.add_argument('-i','--input', action='store', type=str, required=True,help='input file msi')
my_parser.add_argument('-o1','--output1', action='store', type=str, required=True,help='ouput file msi imzml format')


args = my_parser.parse_args()
msi = args.input

output_file = args.output1
profile_to_cent(msi,output_file)


