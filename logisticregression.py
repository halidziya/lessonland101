#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 20:09:57 2021

@author: halidziya
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Model

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize


filename = '/home/halidziya/Downloads/emails.csv'

def logistic(x): # Sigmoid
    return 1/(1+np.exp(-x))

x = np.arange(-10,10,0.1)
plt.plot(x, list(map(logistic, x)))



data = pd.read_csv(filename)

labels = data['spam'].to_numpy()
numeric_data = pd.DataFrame([dict(Counter(t)) for t in data.text]).fillna(0).to_numpy()

x_train, x_test, y_train, y_test = train_test_split(numeric_data, labels)

#Keras
lin = kl.Input((x_train.shape[1],))
out = kl.Dense(1, activation='sigmoid')(lin)
model = Model(lin, out)
model.compile('adam','binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)
print(model.evaluate(x_test, y_test))

#Sklearn
lr = LogisticRegression()
lr.fit(numeric_data, data['spam'])
print(np.mean(lr.predict(x_test)==y_test))

from sklearn.svm import SVC
svm = SVC(probability=True)
svm.fit(x_train, y_train)
pred = svm.predict_proba(x_test)[:,1]

x=[]
y=[]
for treshold in np.arange(0, 1, 0.01):
    positives = (pred>treshold)
    tpr = np.mean(np.sum((positives==True)&(y_test==1))/(np.sum(y_test==1)))
    fpr = np.mean(np.sum((positives==True)&(y_test==0))/(np.sum(y_test==0)))
    x.append(fpr)
    y.append(tpr)
plt.plot(x,y)

#%% Word Level

numeric_data = pd.DataFrame([dict(Counter(word_tokenize(t))) for t in data.text]).fillna(0).to_numpy()
x_train, x_test, y_train, y_test = train_test_split(numeric_data, labels)


lin = kl.Input((x_train.shape[1],))
out = kl.Dense(1, activation='sigmoid')(lin)
model = Model(lin, out)
model.compile('adam','binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)
print(model.evaluate(x_test, y_test))

lr = LogisticRegression()
lr.fit(numeric_data, data['spam'])
print(np.mean(lr.predict(x_test)==y_test))

pred = lr.predict(x_test)
precison = np.sum((pred>0.5)&(y_test==1))/np.sum(pred==1)
recall = np.sum((pred>0.5)&(y_test==1))/np.sum(y_test==1)
f1 = 2*(precison*recall)/(precison+recall)


