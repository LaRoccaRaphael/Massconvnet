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
    
    def __init__(self, path, pre_process_param_name,network_param_name, with_masses=False, mode="train", normalization=False):
        
        # Parameters
        self.path = path
        self.with_masses = with_masses
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
        #df = pd.read_csv(os.path.join(path,"Annot_table.csv"),sep=",", header=0)
        df = pd.read_csv(self.pre_process_param["annot_table"],sep=",", header=0)
        df['initial index'] = np.arange(0,np.shape(df)[0],1)
        df = df.sample(frac = 1,random_state=0)
        
        
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
        
        spec,graph = load_spectrum(path_spectrum,path_graph,self.network_params,self.pre_process_param,self.mode)
        

        # normalize intensity features
        node_features = normalize(torch.clamp(torch.log(torch.from_numpy(spec[:,2])),min=-10).type(torch.float)).unsqueeze(-1)
        #node_features = normalize(torch.clamp(torch.from_numpy(spec[:,2]),min=-10).type(torch.float)).unsqueeze(-1)
        
     
        if self.with_masses:
            
            # normalize mass features
            massrange_diff = self.pre_process_param['mass range'][1]-self.pre_process_param['mass range'][0]
            km = torch.from_numpy((spec[:,0]-(massrange_diff/2))/(massrange_diff/2)).type(torch.float)
            #km = torch.from_numpy(normalize(spec[:,0])).type(torch.float)
            kmd = torch.from_numpy((spec[:,1]+0.5)/0.5).type(torch.float)
            #kmd = torch.from_numpy(normalize(spec[:,1])).type(torch.float)
            node_features = torch.cat((node_features,km.unsqueeze(-1), kmd.unsqueeze(-1)),-1)

        
        # Store data
        #print("shape graph ", np.shape(graph)," ",len(np.shape(graph)))
        #if len(np.shape(graph)) == 0:
        #    print("Setting empty graph")
        #    graph = np.zeros((1,3))

        data = Data(x=node_features,y=torch.from_numpy(np.asarray(self.target_data.loc[index][self.network_params['Annotation name']])), edge_index=torch.transpose(torch.tensor(graph[:,[0,1]]).type(torch.LongTensor),0,1), edge_attr=torch.tensor(graph[:,2]).type(torch.LongTensor).unsqueeze(-1))
        

        return data


    def __len__(self):
        if self.mode=="train":
            return int(np.shape(self.target_data)[0]*self.percent_train)
        elif self.mode == "valid":
            return int(np.shape(self.target_data)[0]*(1-self.percent_train))

        return np.shape(self.target_data)[0]

class MSIDataset(Dataset):
    """
    Dataset loader for the MSI dataset
    - Formats the input data in the graph format
    - Formats the targets for the classification task
    :param path: path to the MSI dataset
    :param spectrum_names: list of spectrums to load
    """

    def __init__(self, path, spectrum_names, with_masses=False, mode="train", normalization=False):

        # Parameters
        self.path = path
        self.spectrum_names = spectrum_names
        self.with_masses = with_masses
        self.mode = mode
        self.normalization=normalization
        self.percent_train = 0.8

        # Load the targets
        target_data = pd.read_csv(os.path.join(path,"Annot_table.csv"),sep="\t", header=0)
        target_MSI_name = target_data.loc[:,"MSI name"].values
        target_MSI_pixel_id = target_data.loc[:,"MSI pixel id"].values
        target_class = target_data.loc[:,"Annotations"].values

        self.num_classes = np.max(target_class)+1

        # Load the adjacency matrix
        adjacency_matrix = np.load(os.path.join(path,"Peaks_adjacency_matrix","adj_mat_highres.npy"),allow_pickle=True)

        # Load the spectrums and associate the correct targets
        self.spectrums = None
        self.targets = None
        for spectrum_name in spectrum_names:
            #tmp_spectrum = torch.clamp(torch.log(torch.transpose(torch.from_numpy(np.load(os.path.join(path,"MSI_datacube",spectrum_name+"_msi.npy"),allow_pickle=True)),0,1)),min=-10).type(torch.float)
            tmp_spectrum = torch.clamp(torch.transpose(torch.from_numpy(np.load(os.path.join(path,"MSI_datacube",spectrum_name+"_msi.npy"),allow_pickle=True)),0,1),min=-10).type(torch.float)
            tmp_target = torch.from_numpy(target_class[np.where(target_MSI_name==spectrum_name)]).type(torch.LongTensor)
            if self.spectrums is not None:
                self.spectrums = torch.cat((self.spectrums,tmp_spectrum),0)
                self.targets = torch.cat((self.targets,tmp_target),0)
            else:
                self.spectrums = tmp_spectrum
                self.targets = tmp_target
        
        # Get the indexes of each class
        self.class_indexes = list()
        for i in np.arange(self.num_classes):
            self.class_indexes.append((self.targets == i).nonzero(as_tuple=True)[0])

            if self.mode=="train":
                self.class_indexes[-1] = self.class_indexes[-1][:int(len(self.class_indexes[-1])*self.percent_train)]
            elif self.mode=="valid":
                self.class_indexes[-1] = self.class_indexes[-1][int(len(self.class_indexes[-1])*self.percent_train):]

        # Load the mass and mass defect
        self.mass = torch.from_numpy(np.load(os.path.join(path,"Peaks_descriptor","KM_ceil.npy"),allow_pickle=True)).type(torch.float)
        self.mass_defect = torch.from_numpy(np.load(os.path.join(path,"Peaks_descriptor","KMD_ceil.npy"),allow_pickle=True)).type(torch.float)

        # Build the Graph
        self.num_features = 1 if not with_masses else 3
        self.num_nodes = adjacency_matrix.shape[0]
        self.num_relations = int(np.max(adjacency_matrix))

        self.edge_index = list()
        self.edge_type = list()
        for i in np.arange(adjacency_matrix.shape[0]):
            for j in np.arange(adjacency_matrix.shape[1]):
                if (adjacency_matrix[i,j] > 0) & (i <j):
                    self.edge_index.append([i,j])
                    self.edge_type.append(adjacency_matrix[i,j]-1)
        self.edge_index = torch.transpose(torch.tensor(self.edge_index).type(torch.LongTensor),0,1)
        self.edge_type = torch.tensor(self.edge_type).type(torch.LongTensor).unsqueeze(-1)

        if self.normalization:

            self.spectrums = normalize(self.spectrums)
            self.mass = normalize(self.mass)
            self.mass_defect = normalize(self.mass_defect)



    def __getitem__(self, index):

        # In train and valid mode
        # Select randomly accross the three classes
        
        if self.mode in ["train", "valid"]:
            # Retrieve the game index and the anchor
            class_selection = random.randint(0, self.num_classes-1)
            index = self.class_indexes[class_selection][random.randint(0,len(self.class_indexes[class_selection])-1)]
        
        node_features = self.spectrums[index].unsqueeze(-1)
        if self.with_masses:
            node_features = torch.cat((node_features,self.mass.unsqueeze(-1), self.mass_defect.unsqueeze(-1)),-1)

        #Store the data for batching
        data = Data(x=node_features,y=self.targets[index], edge_index=self.edge_index, edge_attr=self.edge_type)
        
        return data

    def __len__(self):
        if self.mode=="train":
            return int(self.spectrums.size()[0]*self.percent_train)
        elif self.mode == "valid":
            return int(self.spectrums.size()[0]*(1-self.percent_train))

        return self.spectrums.size()[0]

# Batch the GCN data
def collateGCN(data):
    return Batch.from_data_list(data)

def normalize(data):

    mean = data.mean()
    std = data.std()

    return (data-mean)/std

#def normalize(data):

#    dmax = data.max()
#    dmin = data.min()

#    return (data-dmin)/(dmax - dmin)

def generate_kendrick_param(peaks):
    md = 1
    #md = 14/14.0156500641
    km = peaks*md
    kmd = km - np.ceil(km)
    
    return(km,kmd)

def load_spectrum(path_spectrum,path_graph,signal_degradation_params,pre_process_param,mode):
    
    
    # load original spectrum and graph
    spec = np.load(path_spectrum)
    graph = np.load(path_graph)
    
    applied_degradation = signal_degradation_params['signal degradation']
    
    if  signal_degradation_params['only test'] & (mode != "test"):
        applied_degradation = False
        
    if  signal_degradation_params['only train'] & (mode != "train"):
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
            mass = org_mass + np.random.normal(loc = 0, scale = signal_degradation_params['mass shift param'], size = 1)[0]
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

if __name__ == "__main__":


    dataset = MSIRawDataset("../MSIdataset/", ["mcf7_wi38"], mode="train")

    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=3, shuffle=True,
            num_workers=8, pin_memory=True,collate_fn=collateGCN)

    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
            for i, (data) in t:
                print(data)
