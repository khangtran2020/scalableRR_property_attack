import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CelebA(Dataset):
    def __init__(self, client_df, mode, data_type='embedding', transform=None,
                 data_path='Data/embeddings/', z='Male', y='Eyeglasses', feat_matrix = None, dataset_type='train'):
        self.image_id = client_df['image_id'].values
        self.mode = mode
        self.type = data_type
        self.transform = transform
        self.data_path = data_path
        self.att = client_df[z].values
        self.target = client_df[y].values
        self.feat_matrix = feat_matrix
        self.data_type = dataset_type

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, idx):
        filename = self.image_id[idx]
        att = self.att[idx]
        label = self.target[idx]
        protected_att = torch.Tensor([0]) if att < 0 else torch.Tensor([1])
        target = torch.Tensor([0]) if label < 0 else torch.Tensor([1])
        if self.type == 'embedding':
            if self.data_type == 'train':
                if self.mode == 'clean':
                    img_tensor = torch.load(self.data_path + filename)
                else:
                    img_tensor = torch.from_numpy(self.feat_matrix[idx])
            else:
                img_tensor = torch.load(self.data_path + filename)
        else:
            image = Image.open(self.data_path + filename)
            img_tensor = self.transform(image)
        return img_tensor, protected_att, target
