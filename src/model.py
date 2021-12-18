import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNModel(nn.Module):

    def __init__(self, weights=None, input_size=1, num_relations=1):

        super(GCNModel, self).__init__()

        self.load_weights(weights=weights)

        self.input_size = input_size
        self.num_relations = num_relations

    def load_weights(self, weights=None):

        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, inputs):
        return