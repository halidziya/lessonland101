#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 19:37:41 2021

@author: halidziya
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


filename = '/home/halidziya/Downloads/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv'

data = pd.read_csv(filename)


closings = data['Close'].tolist()
lastvalue = closings[0]
for i in range(len(closings)):
    if not np.isnan(closings[i]):
        lastvalue = closings[i]
    else:
        closings[i] = lastvalue
        

plt.plot(closings)