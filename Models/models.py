import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, AdamW, SGD

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(0, n_layers - 1):
            self.layers.append( nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.n_layers = n_layers
        self.activation = torch.nn.ReLU()
        self.out_dim = output_dim

    def forward(self, x):
        h = x
        for i in range(0, self.n_layers):
            h = self.layers[i](h)
            h = self.activation(h)
        h = torch.sigmoid(self.layers[-1](h)) if self.out_dim == 1 else torch.softmax(self.layers[-1](h), dim=-1)
        return h

class Logit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logit, self).__init__()
        self.out_dim = output_dim
        self.layer_1 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = self.layer_1(x)
        out = torch.sigmoid(x) if self.out_dim == 1 else torch.softmax(x, dim=-1)
        return out


def init_model(args, model_type):
    if model_type == 'normal':
        num_classs = 1 if args.num_class <= 2 else args.num_class
    else:
        num_classs = 1 if args.num_att <= 2 else args.num_att
    model = None
    if args.model_type == 'nn':
        model = NeuralNetwork(input_dim=args.num_feat, hidden_dim=args.hid_dim, output_dim=num_classs, n_layers=args.n_hid)
    elif args.model_type == 'lr':
        model = Logit(input_dim=args.num_feat, output_dim=num_classs)
    return model

def init_optimizer(optimizer_name, model, lr, weight_decay):
    if optimizer_name == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer