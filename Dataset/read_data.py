import os
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Dataset.dataset import CelebA


def read_celeba_csv_by_client_id(args, client_id, df):
    client_df = df[df['client_id'] == client_id].copy().reset_index(drop=True)
    client_df = client_df[['image_id', 'gender', args.target, args.att]]
    return client_df


def init_loader(args, df, mode='train'):
    if args.data_type == 'image':
        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        file_path = 'Data/image_align_celeba/'
    else:
        transform = None
        file_path = 'Data/embeddings/'
    dataset = CelebA(client_df=df, mode=args.mode, data_type=args.data_type, transform=transform, data_path=file_path,
                     z=args.att, y=args.target)
    if mode == 'train':
        bz = min(len(dataset), args.batch_size)
        loader = DataLoader(dataset, batch_size=bz, shuffle=True, drop_last=True)
    elif mode == 'aux':
        bz = min(len(dataset), args.aux_bs)
        loader = DataLoader(dataset, batch_size=bz, shuffle=True, drop_last=True)
    else:
        bz = min(len(dataset), args.batch_size)
        loader = DataLoader(dataset, batch_size=bz, shuffle=True, drop_last=False)
    return loader


def build_aux(args, df):
    # divided half-half for train and aux
    if args.aux_type == 'random':
        temp = df.groupby('client_id').count().sample(frac=1, random_state=args.seed).cumsum()
    elif args.aux_type == 'sort':
        temp = df.groupby('client_id').count().sort_values(by='image_id', ascending=False).cumsum()
    else:
        temp = None
    tr_client_id = list(temp[temp['image_id'] <= 81385].index)
    aux_client_id = list(temp[temp['image_id'] > 81385].index)
    tr_df = df[df['client_id'].isin(tr_client_id)].copy().reset_index(drop=True)
    aux_df = df[df['client_id'].isin(aux_client_id)].copy().reset_index(drop=True)
    return tr_df, aux_df
