import os
import time
import torch
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch
import json
from sklearn.model_selection import StratifiedKFold


class MSIRawDataset(Dataset):
    
    def __init__(self, path, pre_process_param_name,network_param_name, with_masses=False,with_intensity=False,mode="train", normalization=False,random_state=0):
        
        # Parameters
        self.path = path
        self.with_masses = with_masses
        self.with_intensity = with_intensity
        self.mode = mode
        self.normalization=normalization
        self.percent_train = 0.8
        
        # load parameters
        
        self.pre_process_param_json_path = path + '/parameters/pre_processing/' + pre_process_param_name + '.json'
        self.network_param_json_path = path + '/parameters/network/' + network_param_name + '.json'
        
        
        # read parameters from preprocessing
        
        network_params = []
        with open(self.network_param_json_path) as json_file:
            network_params = json.load(json_file)
            
        self.network_params = network_params
        
        
        # read parameters from post processing
        pre_process_param = []
        with open(self.pre_process_param_json_path) as json_file:
            pre_process_param = json.load(json_file)
        
        self.pre_process_param = pre_process_param
        self.spectrum_dir_path = self.pre_process_param['output_dir'] + pre_process_param_name +"/"
        
        # graph parameters
        self.num_relations = len(pre_process_param['mass_diff'])
        if self.network_params['edge index to remove'] != None:
            self.num_relations = self.num_relations - len(self.network_params['edge index to remove'])
        
        self.num_features = 1 if not with_masses else 3
        
        
        # Load sample information
        df = pd.read_csv(self.pre_process_param["annot_table"],sep=",", header=0)
        df['initial index'] = np.arange(0,np.shape(df)[0],1)
        
        if self.mode!="test":
            df = df.sample(frac = 1,random_state=random_state)
        
        
        print(df)
        
        # StratifiedKFold 
        # 'training samples' =="kfold"
        if network_params['training samples'] == "kfold": 
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.network_params['kfold seed'])
            list_train = []
            list_test = []
            
            for train, test in skf.split(df['MSI pixel id'], df[self.network_params['Annotation name']]):
                list_train.append(train)
                list_test.append(test)
                
            train_index_kfold = list_train[self.network_params['kfold K']]
            test_index_kfold = list_test[self.network_params['kfold K']]
            df = pd.concat([df.loc[test_index_kfold],df.loc[train_index_kfold]])
            df['kfold'] =np.concatenate((np.repeat(False,len(test_index_kfold)), np.repeat(True,len(train_index_kfold))))
            

        self.num_classes = np.max(df[self.network_params['Annotation name']])+1
        
        if self.mode=="test":
            self.target_data = df.loc[df[self.network_params['training samples']] ==False].copy()
            
            
        else:
            # Data augmentation
            if network_params['Data augmentation'] >1: 
                n_df = df.copy()
                for i in range(0,network_params['Data augmentation']-1):
                    df = pd.concat([df, n_df], axis=0,ignore_index=True)
            
            self.target_data = df.loc[df[self.network_params['training samples']] ==True].copy()
            
        print(np.unique(self.target_data['initial index']))
            
            
        # Get the indexes of each class
        self.class_indexes = list()
        for i in np.arange(self.num_classes):
            self.class_indexes.append(self.target_data.loc[self.target_data[self.network_params['Annotation name']] == i].index.to_numpy())

            if self.mode=="train":
                self.class_indexes[-1] = self.class_indexes[-1][:int(len(self.class_indexes[-1])*self.percent_train)]
            elif self.mode=="valid":
                self.class_indexes[-1] = self.class_indexes[-1][int(len(self.class_indexes[-1])*self.percent_train):]
            
                
    def __getitem__(self, index):

        # In train and valid mode
        # Select randomly accross the three classes
        index = self.target_data.index[index]
        
        if self.mode in ["train", "valid"]:
            class_selection = random.randint(0, self.num_classes-1)
            index = self.class_indexes[class_selection][random.randint(0,len(self.class_indexes[class_selection])-1)]
        
        # load individual spectrum (km,kmd,intensity)
        # load individual spectrum graph (node1,node2,edgetype)
        path_spectrum = self.spectrum_dir_path + self.target_data.loc[index]['MSI name'] + '/spec_' + str(self.target_data.loc[index]['MSI pixel id']) + '.npy'
        path_graph = self.spectrum_dir_path + self.target_data.loc[index]['MSI name'] + '/graph_' + str(self.target_data.loc[index]['MSI pixel id']) + '.npy'
        
        spec,graph = load_spectrum(path_spectrum,path_graph)
        
        #node_features = torch.from_numpy(np.ones(len(spec[:,2]))).type(torch.float).unsqueeze(-1)

        if self.with_intensity:
            node_features = normalize(torch.clamp(torch.log(torch.from_numpy(spec[:,2])),min=-10).type(torch.float)).unsqueeze(-1)
        else:
            node_features = torch.from_numpy(np.ones(len(spec[:,2]))).type(torch.float).unsqueeze(-1)
            
        
     
        if self.with_masses:
            
            # normalize mass features
            massrange_diff = self.pre_process_param['mass range'][1]-self.pre_process_param['mass range'][0]
            km = torch.from_numpy((spec[:,0]-(massrange_diff/2))/(massrange_diff/2)).type(torch.float)
            #km = torch.from_numpy(normalize(spec[:,0])).type(torch.float)
            kmd = torch.from_numpy((spec[:,1]+0.5)/0.5).type(torch.float)
            #kmd = torch.from_numpy(normalize(spec[:,1])).type(torch.float)
            node_features = torch.cat((node_features,km.unsqueeze(-1), kmd.unsqueeze(-1)),-1)

       

        data = Data(x=node_features,y=torch.from_numpy(np.asarray(self.target_data.loc[index][self.network_params['Annotation name']])), edge_index=torch.transpose(torch.tensor(graph[:,[0,1]]).type(torch.LongTensor),0,1), edge_attr=torch.tensor(graph[:,2]).type(torch.LongTensor).unsqueeze(-1))
        

        return data


    def __len__(self):
        if self.mode=="train":
            return int(np.shape(self.target_data)[0]*self.percent_train)
        elif self.mode == "valid":
            return int(np.shape(self.target_data)[0]*(1-self.percent_train))

        return np.shape(self.target_data)[0]


# Batch the GCN data
def collateGCN(data):
    return Batch.from_data_list(data)

def normalize_tic(data):

    sum_d = data.sum()

    return (data)/sum_d

def normalize(data):

    mean = data.mean()
    std = data.std()

    return (data-mean)/std

def normalize_min_max(data):

    dmax = data.max()
    dmin = data.min()

    return (data-dmin)/(dmax - dmin)

def No_normalize(data):

    return data


def generate_kendrick_param(peaks):
    md = 1
    #md = 14/14.0156500641
    km = peaks*md
    kmd = km - np.ceil(km)
    
    return(km,kmd)

def load_spectrum(path_spectrum,path_graph):
    
    
    # load original spectrum and graph
    spec = np.load(path_spectrum)
    graph = np.load(path_graph)
    graph2 = graph.copy()
    #print(np.shape(graph))
    if np.shape(graph)[0] >0:
        graph = np.concatenate((graph,graph2[:,[1,0,2]]),axis=0)
    else:
        #spec =   np.load("/media/USB2/DL_MASS/DS3/MSI/centroid_data/param_1reduce/161007_WT1S1L1/spec_128.npy")
        #graph =  np.load("/media/USB2/DL_MASS/DS3/MSI/centroid_data/param_1reduce/161007_WT1S1L1/graph_128.npy")
        print("empty graph")
    
    return(spec,graph)

if __name__ == "__main__":


    dataset = MSIRawDataset("../MSIdataset/", ["mcf7_wi38"], mode="train")

    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=3, shuffle=True,
            num_workers=8, pin_memory=True,collate_fn=collateGCN)

    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
            for i, (data) in t:
                print(data)
