from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

import pickle5 as pickle
import pandas as pd
import numpy as np
import os

# path that trained models are saved.
# You can use path below if you didn't change manually save path of "autogluon_regressor.py"
# model_root_dir = "./autogluon_models_full/"
model_root_dir = "/data/jaeyeong/dacon/Image_Quality_Assessment/IQA/DL_ML_hybrid/autogluon_models_full"

data_root_dir = "/data/jaeyeong/dacon/Image_Quality_Assessment/IQA/DL_ML_hybrid/"
# data_root_dir = None

def get_prediction(model_dir, data_dir):
    outs = []
    labels = []

    for md in model_dir:
        model = TabularPredictor.load(os.path.join(model_root_dir, md))

        for d in data_dir:
            with open(os.path.join(data_root_dir, d), "rb") as f:
                features_ = pickle.load(f)
                
            features_ = features_

            features = {}

            for i in range(len(features_)):
                features[features_[i][0]] = (features_[i][1], features_[i][2])
                
            train = []
            targets = []

            for k, v in features.items():
                if len(v[0].shape) > 2:
                    train.append(np.squeeze(v[0], axis=(1, 2)))
                else:
                    train.append(v[0])
                targets.append(v[1])
                
            test_df = pd.DataFrame(np.array(train))
            print(test_df.shape)
            out = model.predict(test_df)
            outs.append(out)
            labels.append(targets)
            print(md, d)
    return outs, labels

predictions = []
labels = []


out, label = get_prediction(['nfnet_f5_train_features', 'nfnet_f5_train_features_flip'], # path for loading model trained by train dataset
                            ['nfnet_f5_test_features.pkl', 'nfnet_f5_test_features_flip.pkl']) # path for loading dataset (test dataset)
predictions.extend(out)
labels.extend(label)

out, label = get_prediction(['vit_h_train_features', 'vit_h_train_features_flip'],
                            ['vit_h_test_features.pkl', 'vit_h_test_features_flip.pkl'])
predictions.extend(out)
labels.extend(label)

out, label = get_prediction(['swinv2_l_1k_train_features', 'swinv2_l_1k_train_features_flip'],
                            ['swinv2_l_1k_test_features.pkl', 'swinv2_l_1k_test_features_flip.pkl'])
predictions.extend(out)
labels.extend(label)

out, label = get_prediction(['swinv2_l_22k_train_features', 'swinv2_l_22k_train_features_flip'],
                            ['swinv2_l_22k_test_features.pkl', 'swinv2_l_22k_test_features_flip.pkl'])
predictions.extend(out)
labels.extend(label)


with open("./mos_prediction.pkl", "wb") as f:
    pickle.dump([predictions, labels], f)
