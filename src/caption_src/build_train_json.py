import json
import pandas as pd
    
from sklearn.model_selection import train_test_split

# transform train dataframe to json (Coco dataset format)

# set dataframe path (train.csv)
# csv_dir = "/data/jaeyeong/dacon/Image_Quality_Assessment/data/train.csv"
csv_dir = None

# set path for save file
# save_dir = "/data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/raw_data/train.json"
save_dir = None

df = pd.read_csv(csv_dir)

name_cap_dict = {}
name_id_dict = {}

for i in range(len(df)):
    img_name = df['img_name'].iloc[i]
    
    if img_name not in name_cap_dict:
        name_cap_dict[img_name] = []
        
    name_cap_dict[img_name].append(df['comments'].iloc[i])
    name_id_dict[img_name] = i
    
train_infos = []
val_infos = []

img_name_set = list(set(df['img_name']))
train_img_names, test_img_names = train_test_split(img_name_set, test_size=0.1, random_state=42, shuffle=True)

for i in train_img_names:
    train_infos.append({"img_id":name_id_dict[i], 
                 "split":"train", 
                 "path":f"/data/jaeyeong/dacon/Image_Quality_Assessment/data/train/{i}",
                 "captions":name_cap_dict[i]}
                 )
for i in test_img_names[:4000]:
    train_infos.append({"img_id":name_id_dict[i], 
                 "split":"val", 
                 "path":f"/data/jaeyeong/dacon/Image_Quality_Assessment/data/train/{i}",
                 "captions":name_cap_dict[i]}
                 )

for i in test_img_names[4000:]:
    train_infos.append({"img_id":name_id_dict[i], 
                 "split":"test", 
                 "path":f"/data/jaeyeong/dacon/Image_Quality_Assessment/data/train/{i}",
                 "captions":name_cap_dict[i]}
                 )


with open(save_dir, "w") as f:
    json.dump(train_infos, f)
    