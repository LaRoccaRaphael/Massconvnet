#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import json
import pandas as pd
 
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

def load_spectrum(spec,graph,signal_degradation_params,pre_process_param,mode):
    
    
    applied_degradation = signal_degradation_params['signal degradation']
    
    if  signal_degradation_params['only test'] & (mode != 0):
        applied_degradation = False
        
    if  signal_degradation_params['only train'] & (mode != 1):
        applied_degradation = False
    
    if applied_degradation :
        print("applied signal degradation")
        
        if signal_degradation_params['spectral resolution param'] >0:
            # shift all the masses from different random values
            org_mass = spec[:,0]
            mass = org_mass + np.random.normal(loc = 0, scale = signal_degradation_params['spectral resolution param'], size = np.shape(org_mass))
            km, kmd = generate_kendrick_param(mass)
            spec[:,0] = km
            spec[:,1] = kmd
            # check if the edge is in the tolerance range
            edge_mass_diff = pre_process_param["mass_diff"]
            edge_mass_diff = np.asarray(edge_mass_diff)
            
            edge_to_keep = np.abs(spec[:,0][graph[:,1]] - spec[:,0][graph[:,0]] - edge_mass_diff[graph[:,2]]) <= pre_process_param["tolerance"]
            graph = graph[edge_to_keep,:]
            
        if signal_degradation_params['mass shift param'] >0:
            # shift all the masses from a random value 
            org_mass = spec[:,0]
            #mass = org_mass + np.random.normal(loc = 0, scale = signal_degradation_params['mass shift param'], size = 1)[0]
            mass = org_mass + signal_degradation_params['mass shift param']
            
            km, kmd = generate_kendrick_param(mass)
            spec[:,0] = km
            spec[:,1] = kmd
            
        if signal_degradation_params['intensity limitation param'] <1:
            # decrease the observed peaks from a given proportion
            index_peak = np.argsort(spec[:,2])[::-1][:np.floor(len(spec[:,2])*signal_degradation_params['intensity limitation param']).astype(int)]
            spec = spec[index_peak,:]
            spec = spec[np.argsort(spec[:,0]),:]

            # create a dictionnary from old peak index to new one according to the sorted spectrum
            keys_list = index_peak[np.argsort(index_peak)] 
            values_list = np.arange(0,len(keys_list),1)
            zip_iterator = zip(keys_list, values_list)
            new_index_dict = dict(zip_iterator)

            # update the edge index 
            edgetokeep = np.zeros(len(graph[:,0]))
            for i in range(0,len(graph[:,0])):
                if (graph[i,0] in new_index_dict) & (graph[i,1] in new_index_dict):
                    edgetokeep[i] = 1
                    graph[i,0] = new_index_dict[graph[i,0]]
                    graph[i,1] = new_index_dict[graph[i,1]]

            graph = graph[edgetokeep.astype(bool),:]
            
        if signal_degradation_params['random peaks removal param'] <1:
            # decrease the observed peaks from a given proportion
            index_peak = np.where(np.random.choice([0, 1], size=(len(spec[:,0]),), p=[1-signal_degradation_params['random peaks removal param'], signal_degradation_params['random peaks removal param']]))[0]
            spec = spec[index_peak,:]
            spec = spec[np.argsort(spec[:,0]),:]

            # create a dictionnary from old peak index to new one according to the sorted spectrum
            keys_list = index_peak[np.argsort(index_peak)] 
            values_list = np.arange(0,len(keys_list),1)
            zip_iterator = zip(keys_list, values_list)
            new_index_dict = dict(zip_iterator)

            # update the edge index 
            edgetokeep = np.zeros(len(graph[:,0]))
            for i in range(0,len(graph[:,0])):
                if (graph[i,0] in new_index_dict) & (graph[i,1] in new_index_dict):
                    edgetokeep[i] = 1
                    graph[i,0] = new_index_dict[graph[i,0]]
                    graph[i,1] = new_index_dict[graph[i,1]]

            graph = graph[edgetokeep.astype(bool),:]
            
            
            
    if signal_degradation_params['edge index to remove'] != None:
        # remove edges and update the graph edge indexes
        
        new_graph = graph.copy()
        index_edge = np.ones(len(graph[:,2]))
        
        for i in signal_degradation_params['edge index to remove']:
            index_edge[graph[:,2] == i] = 0
            new_graph[graph[:,2]>i,2] = graph[graph[:,2]>i,2]-1
            
        graph = new_graph[index_edge.astype(bool),:]
        
    return(spec,graph)

np.random.seed(0)
def rand_bin_array(K, N):
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr

def Iterate_df(network_params,pre_process_param,pre_process_param_name,network_param_name):
    
    file = pre_process_param['output_dir'] + pre_process_param_name +"/"
    df = pd.read_csv(pre_process_param['annot_table'],sep=",", header=0)
    
    ## applied signal mixing 
    if network_params["Signal mixing"] > 0:
        
        rd = 1
        
        prop = network_params["Signal mixing"]
        max_peaks = pre_process_param['max_peaks']
        mass_diff = pre_process_param['mass_diff']
        tolerance = pre_process_param['tolerance']
        
        if network_params['only test']:
            sub_df = df.loc[(df["train"] == 0)].copy()
            unmodified_df = df.loc[(df["train"] == 1)].copy()
            
        if network_params['only train']:
            sub_df = df.loc[(df["train"] == 1)].copy()
            unmodified_df = df.loc[(df["train"] == 0)].copy()
            
        if network_params['only train'] & network_params['only test']:
            sub_df = df.copy()
            unmodified_df = pd.DataFrame()
        
        for annot in np.unique(df["Annotations"]):
                           
            Samples_class_0 = sub_df.loc[(df["Annotations"] == annot)].copy().reset_index()
            Samples_class_1 = sub_df.loc[(df["Annotations"] != annot)].copy().sample(frac=1,random_state=rd).reset_index()
            Samples_class_0["Sample to modified"] = rand_bin_array(int(np.shape(Samples_class_0)[0]*0.5),np.shape(Samples_class_0)[0])
            
            unmodified_Samples_class_0 = unmodified_df.loc[(df["Annotations"] == annot)].copy().reset_index()
            unmodified_Samples_class_1 = unmodified_df.loc[(df["Annotations"] != annot)].copy().sample(frac=1,random_state=rd).reset_index()
            unmodified_Samples_class_0["Sample to modified"] = rand_bin_array(int(np.shape(unmodified_Samples_class_0)[0]*0.5),np.shape(unmodified_Samples_class_0)[0])
            
            for index, row in Samples_class_0.iterrows():
                
                path = file + row['MSI name'] +"/spec_" + str(row['MSI pixel id']) + ".npy"
                spec1 = np.load(path)
                
                
                row2 = Samples_class_1.loc[index]
                path = file + row2['MSI name'] +"/spec_" + str(row2['MSI pixel id']) + ".npy"
                spec2 = np.load(path)
                spec2[:,2] = spec2[:,2]*prop*row["Sample to modified"]
                
                spec = np.concatenate((spec1,spec2),axis=0)
                #spec = spec[np.argsort(spec[:,2])[::-1][:max_peaks],:]
                spec = spec[np.argsort(spec[:,0]),:]
                
                spectrum_edge_param = generate_graph_from_spectrum(spec,mass_diff,tolerance)
                
                np.save(pre_process_param['output_dir']+ network_param_name + "/" +row['MSI name']+"/spec_" + str(row['MSI pixel id']) + ".npy",spec)
                np.save(pre_process_param['output_dir']+ network_param_name + "/" +row['MSI name']+"/graph_" + str(row['MSI pixel id']) + ".npy",spectrum_edge_param )
            
            for index, row in unmodified_Samples_class_0.iterrows():
                
                path = file + row['MSI name'] +"/spec_" + str(row['MSI pixel id']) + ".npy"
                spec1 = np.load(path)
                
                row2 = unmodified_Samples_class_1.loc[index]
                path = file + row2['MSI name'] +"/spec_" + str(row2['MSI pixel id']) + ".npy"
                spec2 = np.load(path)
                spec2[:,2] = spec2[:,2]*0
                
                spec = np.concatenate((spec1,spec2),axis=0)
                spec = spec[np.argsort(spec[:,0]),:]
                spectrum_edge_param = generate_graph_from_spectrum(spec,mass_diff,tolerance)
                
                #path = file + row['MSI name'] +"/graph_" + str(row['MSI pixel id']) + ".npy"
                #graph = np.load(path)
                
                np.save(pre_process_param['output_dir']+ network_param_name + "/" +row['MSI name']+"/spec_" + str(row['MSI pixel id']) + ".npy",spec)
                np.save(pre_process_param['output_dir']+ network_param_name + "/" +row['MSI name']+"/graph_" + str(row['MSI pixel id']) + ".npy",spectrum_edge_param)

         
    ## applied other           
    else:
        
        for index, row in df.iterrows():

            print(str(row['MSI pixel id']),row['train'])
            path = file + row['MSI name'] +"/spec_" + str(row['MSI pixel id']) + ".npy"
            spec = np.load(path)

            path = file + row['MSI name'] +"/graph_" + str(row['MSI pixel id']) + ".npy"
            graph = np.load(path)

            mode = row['train']


            #plt.scatter(spec[:,0],spec[:,1],1)
            spec,graph = load_spectrum(spec,graph,network_params,pre_process_param,mode)
            #plt.scatter(spec[:,0],spec[:,1],1)
            #plt.show()

            np.save(pre_process_param['output_dir']+ network_param_name + "/" +row['MSI name']+"/spec_" + str(row['MSI pixel id']) + ".npy",spec)
            np.save(pre_process_param['output_dir']+ network_param_name + "/" +row['MSI name']+"/graph_" + str(row['MSI pixel id']) + ".npy",graph )
 
    

# Load the arguments
my_parser = argparse.ArgumentParser(allow_abbrev=False)
my_parser.add_argument('-p','--path', action='store', type=str, required=True,help='path to the folder')
my_parser.add_argument('-pp','--previous_param', action='store', type=str, required=True,help='name of the json file for the previous parameters')
my_parser.add_argument('-np','--new_param', action='store', type=str, required=True,help='name of the json file for the new parameters')


args = my_parser.parse_args()


path = args.path
pre_process_param_name = args.previous_param
network_param_name = args.new_param

pre_process_param_json_path = path + '/parameters/pre_processing/' + pre_process_param_name + '.json'
network_param_json_path = path + '/parameters/network/' + network_param_name + '.json'


network_params = []
with open(network_param_json_path) as json_file:
    network_params = json.load(json_file)
            
   
        
# read parameters from post processing
pre_process_param = []
with open(pre_process_param_json_path) as json_file:
    pre_process_param = json.load(json_file)
        

file = pre_process_param['output_dir'] + pre_process_param_name +"/"


# create folder
os.mkdir(pre_process_param['output_dir']+ network_param_name)

df = pd.read_csv(pre_process_param['annot_table'],sep=",", header=0)

for msi_n in np.unique(df["MSI name"]):
    os.mkdir(pre_process_param['output_dir']+ network_param_name+"/"+msi_n)
    
# generate the new datasets 
Iterate_df(network_params,pre_process_param,pre_process_param_name,network_param_name)


# save new preprocessing parameters
json_object = json.dumps(pre_process_param,indent=4)

with open(path +"parameters/pre_processing/"+network_param_name+".json", "w") as outfile:
    outfile.write(json_object)