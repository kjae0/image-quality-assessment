import os
import argparse
import pandas as pd
import numpy as np
import pickle5 as pickle

import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from torchvision.transforms import transforms
from models import build_model
from dataset import build_dataset

# set image path for load_img function
# set dataframe path : csv_dir
# set image folder path : root_dir 
# set save path : save_dir
# 518 for vit H

# Model weight path (Swin transformer V2 Large)
# You can download it from "https://github.com/microsoft/Swin-Transformer"
# pretrained from imagenet-22k
# image size 192 22k swin

# Model weight path (Swin transformer V2 Large)
# You can download it from "https://github.com/microsoft/Swin-Transformer"
# pretrained from imagenet-1k
# image size 384 1k swin

# Model weight path (NFNet F5)
# You can download it from "https://github.com/google-deepmind/deepmind-research/tree/master/nfnets"
# pretrained from imagenet
# image size 544 nfnet f5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--nf_weight_dir", type=str, default=None)
    parser.add_argument("--state_dict_dir", type=str, default=None)
    args = parser.parse_args()

    # set dataframe path
    # train -> train.csv
    # test -> test.csv
    # csv_dir = "/data/jaeyeong/dacon/Image_Quality_Assessment/data/train.csv"

    model = build_model(args.model_name)
    
    if args.state_dict_dir:
        model.load_state_dict(torch.load(args.state_dict_dir)['model'])
    
    model = nn.DataParallel(model)
    model = model.to('cuda')
    model.eval()

    dl, img_score_pair_lst = build_dataset(args.csv_dir, args.img_size, flip=args.flip, nf_weight_dir=args.nf_weight_dir)
    outputs = []
    image_names = []

    avg_pool = nn.AdaptiveAvgPool2d(1)
    with torch.no_grad():
        for imgs, targets in tqdm(dl, total=len(dl), ncols=60):
            imgs = imgs.to('cuda')
            out = model(imgs)
            outputs.append(out.cpu())
    outputs = torch.concat(outputs, dim=0)

    output_dump = []
    image_names = [i[0] for i in img_score_pair_lst]
    targets = [i[1] for i in img_score_pair_lst]

    assert len(outputs) == len(image_names) == len(targets)

    for i in range(len(outputs)):
        output_dump.append((image_names[i], np.array(outputs[i]), targets[i]))

    with open(args.save_dir, "wb") as f:
        pickle.dump(output_dump, f)
