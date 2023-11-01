import os
import pandas as pd
import numpy as np
import pickle5 as pickle

import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from torchvision.transforms import transforms
from swin_models import SwinTransformerV2

# set image path for load_img function
# set dataframe path : csv_dir
# set image folder path : root_dir 
# set model weight path : model_dir


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, targets):
        self.imgs = imgs
        self.targets = targets
        self.transform = transforms.Compose([
                            transforms.Resize((192, 192)),
                            # transforms.RandomHorizontalFlip(1),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        return self.transform(load_img(self.imgs[index])), self.targets[index]

def load_img(img_name):
    # f-string for image path
    # CONVERTING TO RGB CHANNEL IS REQUIRED
    img = Image.open(f"/data/jaeyeong/dacon/Image_Quality_Assessment/data/train/{img_name}.jpg").convert('RGB')
    # img = Image.open().convert('RGB')
    return img

# set dataframe path
# train -> train.csv
# test -> test.csv
# csv_dir = "/data/jaeyeong/dacon/Image_Quality_Assessment/data/train.csv"
csv_dir = None

# image folder path
# root_dir = "/data/jaeyeong/dacon/Image_Quality_Assessment/data/train"
root_dir = None

# Model weight path (Swin transformer V2 Large)
# You can download it from "https://github.com/microsoft/Swin-Transformer"
# pretrained from imagenet-22k
# model_dir = "/data/jaeyeong/dacon/Image_Quality_Assessment/IQA/swinv2_pretrained/swinv2_L.pth"
model_dir = None

# save_dir = "./swinv2_l_22k_test_features.pkl"
save_dir = None

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

model = SwinTransformerV2()
sd = torch.load(model_dir)
model.load_state_dict(sd['model'])

img_score_pair_lst = [(k, v) for k, v in img_score_pairs.items()]
    
model = model.to('cuda')
model.eval()

outputs = []
image_names = []

dset = CustomDataset(imgs=[i[0] for i in img_score_pair_lst],
                     targets=[i[1] for i in img_score_pair_lst])

dl = torch.utils.data.DataLoader(
    dset,
    batch_size=120,
    shuffle=False,
    drop_last=False,
    num_workers=16
)
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

with open(save_dir, "wb") as f:
    pickle.dump(output_dump, f)
