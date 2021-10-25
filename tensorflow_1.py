#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 20:06:46 2021

@author: halidziya
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

v = 0 # number
vec = [1,2,3] # d
mat = [[0,1], [0,1]] # d1xd2
ten = [[[0,1], [0,1]],[[0,1], [0,1]]] # d1xd2xd3
#...


t = tf.convert_to_tensor([[1,1],[1,1]])

# n x d  X d x m => n x m
# f(w'x) = y
# d -> w -> d
# x_1 * w_1 + x_2 * w_2
# wx + b

r = tf.matmul(t, tf.transpose(t))
print(r)
r.numpy()

print(t + t)


data = np.random.multivariate_normal([0,0], [[0.55,0.45],[0.55,0.45]], 1000)
x = data[:,0]
y = data[:,1]


# Model definition
lin = kl.Input((1,))
out = kl.Dense(1)(lin)
model = Model(lin, out)
model.compile('Adam', 'mse')

model.fit(x, y, epochs=200)
plt.plot(model.history.history['loss'])
plt.show()

plt.scatter(x, y, s=1)
x_ticks = np.arange(-2, 2, 0.1)
y_ticks = model.predict(x_ticks)
plt.plot(x_ticks, y_ticks, c='orange')


y - model.predict(x)
outliers = np.abs(y - model.predict(x)[:,0])>1
plt.scatter(x[outliers], y[outliers])