from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from scipy.stats import pearsonr, spearmanr

import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle5 as pickle


def train(data_dir):
    x_train, x_val, y_train, y_val = [], [], [], []
    
    for d in data_dir:
        with open(d, "rb") as f:
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
            
        x_train_, x_test_, y_train_, y_test_ = train_test_split(train, targets, test_size=0.10, random_state=42, shuffle=True)
        
        x_train.extend(x_train_)
        x_val.extend(x_test_)
        y_train.extend(y_train_)
        y_val.extend(y_test_)

    save_name = data_dir[0]
    model = TabularPredictor(label='target', problem_type='regression', eval_metric='pearsonr', path=f"./autogluon_models_full/{save_name}")
    
    train_df = pd.DataFrame(np.array(x_train))
    val_df = pd.DataFrame(np.array(x_val))
    train_df['target'] = pd.Series(y_train)
    val_df['target'] = pd.Series(y_val)
    
    model.fit(train_df, tuning_data=val_df)
            #   , time_limit=100)

    pred_vit = model.predict(val_df)
    return pred_vit, val_df['target']

if __name__ == "__main__":
    data_lst = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    
    data_lst.append([args.data])
    
    predictions = []
    for data in data_lst:
        pred, y_test = train(data)
        predictions.append(pred)
        print(pred.shape)

    ensemble = predictions[0]
    for i in range(1, len(predictions)):
        ensemble += predictions[i]
        
    ensemble /= len(predictions)

    print(pearsonr(ensemble, y_test), spearmanr(ensemble, y_test))
