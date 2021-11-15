#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 20:15:44 2021

@author: halidziya
"""

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
y = iris.target>0
x = iris.data[:,0:2]
plt.scatter(x[:,0], x[:,1], c=y)



x_test = []
for i in np.arange(4, 8, 0.02):
    for j in np.arange(2, 5, 0.02):
        x_test.append([i, j])

x_test = np.array(x_test)


position = []
for query in x_test:
    position.append(np.argmin(np.linalg.norm(x - query, axis=1)))
    
y_test = y[position]


plt.scatter(x_test[:,0], x_test[:,1], c=y_test, s=1);plt.scatter(x[:,0], x[:,1], c=y)
plt.title('1-nearest neighboor')




y_test2 = []
for query in x_test:
    y_test2.append(np.mean(y[np.argsort(np.linalg.norm(x - query, axis=1))[:5]]) > 0.5)
    
    
plt.scatter(x_test[:,0], x_test[:,1], c=y_test2, s=1);plt.scatter(x[:,0], x[:,1], c=y)
plt.title('5-nearest neighboor')



sigma = 0.1
prob = []
for query in x_test:
    prob.append(np.sum(np.exp(-np.linalg.norm(x - query, axis=1)**2/(2*sigma))))


plt.scatter(x_test[:,0], x_test[:,1], c=prob);plt.scatter(x[:,0], x[:,1], c=y, s=2)




prob = []
for query in x_test:
    p = np.exp(-np.linalg.norm(x - query, axis=1)**2/(0.2))
    prob.append(np.sum(y*(p/sum(p))))
    
plt.scatter(x_test[:,0], x_test[:,1], c=np.array(prob)>0.5);plt.scatter(x[:,0], x[:,1], c=y, s=10)