import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import RGCNConv
from torch_geometric.nn import global_max_pool

class GCNModel(nn.Module):

    def __init__(self, weights=None, input_size=1, num_relations=1, num_classes=3, multiplier=1):

        super(GCNModel, self).__init__()

        # Load the network pre-trained weights if required
        self.load_weights(weights=weights)

        # Network variables
        self.input_size = input_size
        self.num_relations = num_relations
        self.num_classes = num_classes
        # Definition of the GCN layers
        self.rgcnconv_1 = RGCNConv(self.input_size, 8*multiplier, self.num_relations)
        self.rgcnconv_2 = RGCNConv(8*multiplier, 16*multiplier, self.num_relations)

        # Define the classification head
        self.batchnorm = torch.nn.BatchNorm1d(16*multiplier)
        self.linear_1 = torch.nn.Linear(16*multiplier, 8*multiplier)
        self.linear_2 = torch.nn.Linear(8*multiplier, self.num_classes)

    def load_weights(self, weights=None):

        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, data):

        # Retrieve the different parts
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr[:,0], data.batch

        # GCN part
        x = F.relu(self.rgcnconv_1(x,edge_index,edge_type))
        x = F.relu(self.rgcnconv_2(x,edge_index,edge_type))
        x = global_max_pool(x, batch)
        #x =  global_mean_pool(x, batch)
        
        # Classification Head
        #x = self.batchnorm(x)
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)

        return x
    
class GCN_branch_Model(nn.Module):

    def __init__(self, weights=None, input_size=1, num_relations=1, num_classes=3, multiplier=1):

        super(GCNModel, self).__init__()

        # Load the network pre-trained weights if required
        self.load_weights(weights=weights)

        # Network variables
        self.input_size = input_size
        self.num_relations = num_relations
        self.num_classes = num_classes
        
        multiplier=4

        # Definition of the GCN layers
        self.num_branches = 4
        self.rgcnconv = list()
        for i in np.arange(self.num_branches):
            self.rgcnconv_1 = RGCNConv(self.input_size, 8*multiplier, self.num_relations)
            self.rgcnconv_2 = RGCNConv(8*multiplier, 16*multiplier, self.num_relations)
            self.rgcnconv.append([self.rgcnconv_1, self.rgcnconv_2])

        # Define the classification head
        self.batchnorm = torch.nn.BatchNorm1d(16*multiplier*self.num_branches)
        self.linear_1 = torch.nn.Linear(16*self.num_branches*multiplier, 8*self.num_branches*multiplier)
        self.linear_2 = torch.nn.Linear(8*self.num_branches*multiplier, self.num_classes)

    def load_weights(self, weights=None):

        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, data):

        # Retrieve the different parts
        inputs, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr[:,0], data.batch

        # GCN part
        features = None
        for i in np.arange(self.num_branches):
            x = F.relu(self.rgcnconv_1(inputs,edge_index,edge_type))
            x = F.relu(self.rgcnconv_2(x,edge_index,edge_type))
            x = global_max_pool(x, batch)
            if i == 0:
                features = x
            else:
                features = torch.cat((features,x),dim=-1)
        
        
        # Classification Head
        x = self.batchnorm(features)
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)

        return x

if __name__=="__main__":

    from dataloader import MSIDataset, collateGCN
    from tqdm import tqdm


    dataset = MSIDataset("../MSIdataset/", ["mcf7_wi38"], mode="train",with_masses=True)

    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=3, shuffle=True,
            num_workers=1, pin_memory=True,collate_fn=collateGCN)

    model = GCNModel(weights=None, input_size=dataset.num_features,num_relations=dataset.num_relations, num_classes=dataset.num_classes, multiplier=1).cuda()

    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
            for i, (data) in t:
                print(data)
                print(data.x.shape)
                output = model(data.cuda())
                print(output.shape)
                