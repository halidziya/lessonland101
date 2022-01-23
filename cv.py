# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 16:08:41 2022

@author: halid
"""
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

# Load data

train_data, test_data = train_test_split(data)

result = {}
for c in train_data.columns:
    result[c] = normalized_mutual_info_score(train_data['age'], train_data[c])
    
    
np.array(list(result.keys()))[np.argsort(np.array(list(result.values())))]



from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
model = LinearRegression()
x_train = train_data.loc[ : , data.columns != 'age'].to_numpy()
y_train = train_data['age'].to_numpy()


model_selection_table = np.zeros((x_train.shape[1],5))
for j in range(x_train.shape[1]): # Drop the feature
    for i in range(5): # CV
        start = (i*x_train.shape[0]//5)
        end  = ((i+1)*x_train.shape[0]//5)
        idx = np.arange(0,x_train.shape[0])
        valid_idx = (start <= idx) & (idx<end)
        feature_select = np.ones(x_train.shape[1], dtype=bool)
        feature_select[j] = 0
        x_valid = x_train[valid_idx][:,feature_select]
        y_valid = y_train[valid_idx]
        model.fit(x_train[~valid_idx][:,feature_select], y_train[~valid_idx])
        score = np.mean((np.square(model.predict(x_valid)-y_valid)))
        print(score)
        model_selection_table[j, i] = score



