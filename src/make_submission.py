import numpy as np
import pandas as pd
import pickle5 as pickle

with open("./inference_full.pkl", 'rb') as f:
    output = pickle.load(f)
    
ensemble = np.zeros(output[0][0].shape)

for out in output[0]:
    ensemble += out

ensemble /= 16
    
submission = pd.read_csv("/data/jaeyeong/dacon/Image_Quality_Assessment/data/sample_submission.csv")
submission['mos'] = pd.Series(ensemble)

comments = pd.read_csv("/data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/comments3.csv")

name_id_mapper = {}

for i in range(len(submission)):
    name_id_mapper[submission['img_name'].iloc[i]] = i

for i in range(len(comments)):
    submission['comments'].iloc[name_id_mapper[comments['img_name'].iloc[i].split(".")[0]]] = comments['comments'].iloc[i]

submission.to_csv("./final_submission.csv", index=False)
