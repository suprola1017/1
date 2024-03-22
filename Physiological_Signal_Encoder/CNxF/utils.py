import torch
import os
from Physiological_Signal_Encoder.CNxF.dataset import Multimodal_Datasets

def __init__(*args):
    return

def get_data(args, dataset, split='train'):
    data_path = os.path.join(args.data_path, dataset) + f'_{split}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path) ###
    return data


def save_load_name(args, name=''):
    load_name = name + '_' + args.model
    return load_name


def save_model(args, model, name=''):
    if not os.path.exists('A_Multi_Fusion_Data/'):
        os.makedirs('A_Multi_Fusion_Data/')
    name = save_load_name(args, name)
    torch.save(model, f'A_Multi_Fusion_Data/{args.name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'A_Multi_Fusion_Data/{args.name}.pt')
    return model
