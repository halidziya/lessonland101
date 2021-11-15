#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 11:05:20 2021

@author: halidziya
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import numpy as np
import matplotlib.pyplot as plt

model = ResNet50()


sample = np.random.rand(1, 224,224,3)
noisef = lambda x: 0.8*x+np.random.rand(1, 224,224,3)/5

prior_score = model.predict(sample)[0][0]
for i in range(1000000):
    new_sample = noisef(sample) 
    new_score = model.predict(new_sample)[0][0]
    if (new_score > prior_score):
        sample = new_sample
        prior_score = new_score
    print(prior_score)
    if (i%1000):
        plt.imshow(sample[0])
        plt.show()