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

class IQN_Policy(nn.Module):
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

        self.K = 32 # number of quantiles in output
        self.n = 64 # number of cosine features

        # quantile encoder
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n)]).view(1,1,self.n).to(device)
        self.cos_embedding = nn.Linear(self.n,self.concat_feature_dimension)

        # hidden layers
        self.hidden_layer = nn.Linear(self.concat_feature_dimension, hidden_dimension)
        self.hidden_layer_2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.output_layer = nn.Linear(hidden_dimension, action_size)

        # temp for reproducibility
        # self.taus = None

    def calc_cos(self, batch_size, num_tau=8, cvar=1.0):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        # temp for reproducibility
        # if self.taus is None:
        #     t = np.arange(0.05,1,0.03).astype(np.float32)
        #     self.taus = torch.from_numpy(t).to(self.device).unsqueeze(-1) # (batch_size, n_tau, 1) for broadcast
        # taus = self.taus * cvar

        taus = torch.rand(batch_size,num_tau).to(self.device).unsqueeze(-1)

        # distorted quantile sampling
        taus = taus * cvar

        cos = torch.cos(taus * self.pis)
        assert cos.shape == (batch_size, num_tau, self.n), "cos shape is incorrect"
        return cos, taus
    
    def forward(self, x, num_tau=8, cvar=1.0):
        batch_size = x.shape[0]

        self_states = x[:,:self.self_dimension]
        static_states = x[:,self.self_dimension:self.self_dimension+self.static_dimension]
        dynamic_states = x[:,self.self_dimension+self.static_dimension:]

        # encode observations as features
        self_features = self.self_encoder(self_states)
        static_features = self.static_encoder(static_states)
        dynamic_features = self.dynamic_encoder(dynamic_states)
        features = torch.cat((self_features,static_features,dynamic_features),1)

        # encode quantiles as features
        cos, taus = self.calc_cos(batch_size, num_tau, cvar)
        cos = cos.view(batch_size*num_tau, self.n)
        cos_features = relu(self.cos_embedding(cos)).view(batch_size,num_tau,self.concat_feature_dimension)

        # pairwise product of the input feature and cosine features
        features = (features.unsqueeze(1) * cos_features).view(batch_size*num_tau,self.concat_feature_dimension)
        
        features = relu(self.hidden_layer(features))
        features = relu(self.hidden_layer_2(features))
        quantiles = self.output_layer(features)
        
        return quantiles.view(batch_size,num_tau,self.action_size), taus
    
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
        torch.save(self.state_dict(),os.path.join(directory,f"network_params.pth"))
        
        # save constructor parameters
        with open(os.path.join(directory,f"constructor_params.json"),mode="w") as constructor_f:
            json.dump(self.get_constructor_parameters(),constructor_f)

    @classmethod
    def load(cls,directory,device="cpu"):
        # load network parameters
        model_params = torch.load(os.path.join(directory,"network_params.pth"),
                                  map_location=device)

        # load constructor parameters
        with open(os.path.join(directory,"constructor_params.json"), mode="r") as constructor_f:
            constructor_params = json.load(constructor_f)
            constructor_params["device"] = device

        model = cls(**constructor_params)
        model.load_state_dict(model_params)
        model.to(device)

        return model