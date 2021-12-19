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

class MSIDataset(Dataset):
    """
    Dataset loader for the MSI dataset
    - Formats the input data in the graph format
    - Formats the targets for the classification task
    :param path: path to the MSI dataset
    :param spectrum_names: list of spectrums to load
    """

    def __init__(self, path, spectrum_names, with_masses=False, mode="train"):

        # Parameters
        self.path = path
        self.spectrum_names = spectrum_names
        self.mode = mode
        self.with_masses = with_masses
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
            tmp_spectrum = torch.clamp(torch.log(torch.transpose(torch.from_numpy(np.load(os.path.join(path,"MSI_datacube",spectrum_name+"_msi.npy"),allow_pickle=True)),0,1)),min=-10).type(torch.float)
            tmp_target = torch.from_numpy(target_class[np.where(target_MSI_name==spectrum_name)]).type(torch.LongTensor)
            if self.spectrums is not None:
                self.spectrums = torch.cat((tmp_spectrum,self.spectrums),0)
                self.targets = torch.cat((tmp_target,self.targets),0)
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
                if adjacency_matrix[i,j] > 0:
                    self.edge_index.append([i,j])
                    self.edge_type.append(adjacency_matrix[i,j]-1)
        self.edge_index = torch.transpose(torch.tensor(self.edge_index).type(torch.LongTensor),0,1)
        self.edge_type = torch.tensor(self.edge_type).type(torch.LongTensor).unsqueeze(-1)


    def __getitem__(self, index):

        # In train and valid mode
        # Select randomly accross the three classes
        if self.mode in ["train", "valid"]:
            # Retrieve the game index and the anchor
            class_selection = random.randint(0, self.num_classes-1)
            index = random.randint(0,len(self.class_indexes[class_selection])-1)
        
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



if __name__ == "__main__":


    dataset = MSIDataset("../MSIdataset/", ["mcf7_wi38"], mode="train")

    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=3, shuffle=True,
            num_workers=8, pin_memory=True,collate_fn=collateGCN)

    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
            for i, (data) in t:
                print(data)
