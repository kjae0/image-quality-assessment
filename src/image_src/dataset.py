import os
import pandas as pd
import numpy as np
import pickle5 as pickle

import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from torchvision.transforms import transforms

def build_dataset(csv_dir, img_size, flip=False):
    # image folder path
    # root_dir = "/data/jaeyeong/dacon/Image_Quality_Assessment/data/train"        
    df = pd.read_csv(csv_dir)

    # if csv file is for test, add mos column
    if 'mos' not in df.columns:
        df['mos'] = pd.Series([0 for _ in range(len(df))])
        
    img_score_pairs = {}

    for i in range(len(df)):
        if df['img_name'].iloc[i] in img_score_pairs:
            continue
        else:
            img_score_pairs[df['img_name'].iloc[i]] = df['mos'].iloc[i]
    img_score_pair_lst = [(k, v) for k, v in img_score_pairs.items()]

    dset = CustomDataset(imgs=[i[0] for i in img_score_pair_lst],
                        targets=[i[1] for i in img_score_pair_lst],
                        img_size=img_size,
                        flip=flip)

    dl = torch.utils.data.DataLoader(
        dset,
        batch_size=36,
        shuffle=False,
        drop_last=False,
        num_workers=16
    )
    return dl, img_score_pair_lst

def load_img(img_name):
    # f-string for image path
    # CONVERTING TO RGB CHANNEL IS REQUIRED
    # img = Image.open(f"/data/jaeyeong/dacon/Image_Quality_Assessment/data/train/{img_name}.jpg").convert('RGB')
    img = Image.open(img_name).convert('RGB')
    return img


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, targets, img_size, flip=False):
        self.imgs = imgs
        self.targets = targets
        if flip:
            self.transform = transforms.Compose([
                                transforms.Resize((img_size, img_size)),
                                transforms.RandomHorizontalFlip(1),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        else:
            self.transform = transforms.Compose([
                                transforms.Resize((img_size, img_size)),
                                # transforms.RandomHorizontalFlip(1),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        return self.transform(load_img(self.imgs[index])), self.targets[index]
    
