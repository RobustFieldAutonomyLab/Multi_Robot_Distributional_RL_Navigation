import torch
import torch.nn as nn
import numpy as np
import os
import json
from torch.nn.functional import softmax,relu

def encoder(input_dimension,output_dimension):
    l1 = nn.Linear(input_dimension,output_dimension)
    l2 = nn.ReLU()
    model = nn.Sequential(l1, l2)
    return model

class DQN_Policy(nn.Module):
    def __init__(self,
                 self_dimension,
                 static_dimension,
                 dynamic_dimension,
                 self_feature_dimension,
                 static_feature_dimension,
                 dynamic_feature_dimension,
                 hidden_dimension,
                 action_size,
                 device='cpu',
                 seed=0
                 ):
        super().__init__()

        self.self_dimension = self_dimension
        self.static_dimension = static_dimension
        self.dynamic_dimension = dynamic_dimension
        self.self_feature_dimension = self_feature_dimension
        self.static_feature_dimension = static_feature_dimension
        self.dynamic_feature_dimension = dynamic_feature_dimension
        self.concat_feature_dimension = self_feature_dimension + static_feature_dimension + dynamic_feature_dimension
        self.hidden_dimension = hidden_dimension
        self.action_size = action_size
        self.device = device
        self.seed_id = seed
        self.seed = torch.manual_seed(seed)

        self.self_encoder = encoder(self_dimension,self_feature_dimension)
        self.static_encoder = encoder(static_dimension,static_feature_dimension)
        self.dynamic_encoder = encoder(dynamic_dimension,dynamic_feature_dimension)

        # hidden layers
        self.hidden_layer = nn.Linear(self.concat_feature_dimension, hidden_dimension)
        self.hidden_layer_2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.output_layer = nn.Linear(hidden_dimension, action_size)

    def forward(self, x):

        self_states = x[:,:self.self_dimension]
        static_states = x[:,self.self_dimension:self.self_dimension+self.static_dimension]
        dynamic_states = x[:,self.self_dimension+self.static_dimension:]

        # encode observations as features
        self_features = self.self_encoder(self_states)
        static_features = self.static_encoder(static_states)
        dynamic_features = self.dynamic_encoder(dynamic_states)
        features = torch.cat((self_features,static_features,dynamic_features),1)
        
        features = relu(self.hidden_layer(features))
        features = relu(self.hidden_layer_2(features))
        output = self.output_layer(features)
        
        return output
    
    def get_constructor_parameters(self):       
        return dict(self_dimension = self.self_dimension,
                    static_dimension = self.static_dimension,
                    dynamic_dimension = self.dynamic_dimension,
                    self_feature_dimension = self.self_feature_dimension,
                    static_feature_dimension = self.static_feature_dimension,
                    dynamic_feature_dimension = self.dynamic_feature_dimension,
                    hidden_dimension = self.hidden_dimension,
                    action_size = self.action_size,
                    seed = self.seed_id)
    
    def save(self,directory):
        # save network parameters
        torch.save(self.state_dict(),os.path.join(directory,f"dqn_network_params.pth"))
        
        # save constructor parameters
        with open(os.path.join(directory,f"dqn_constructor_params.json"),mode="w") as constructor_f:
            json.dump(self.get_constructor_parameters(),constructor_f)

    @classmethod
    def load(cls,directory,device="cpu"):
        # load network parameters
        model_params = torch.load(os.path.join(directory,"dqn_network_params.pth"),
                                  map_location=device)

        # load constructor parameters
        with open(os.path.join(directory,"dqn_constructor_params.json"), mode="r") as constructor_f:
            constructor_params = json.load(constructor_f)
            constructor_params["device"] = device

        model = cls(**constructor_params)
        model.load_state_dict(model_params)
        model.to(device)

        return model