# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 18:55:26 2022

@author: halid
"""

import tensorflow
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# [1, 0, 0  .... ], [0, 1, 0  .... ]
# MLP
y_train = to_categorical(y_train)

lin = Input((28,28))
lf = Flatten()(lin)
hidden = Dense(64, activation='relu')(lf)
hidden2 = Dense(64, activation='relu')(hidden)
dense = Dense(10, activation='softmax')(hidden2)

model = Model(lin, dense)

model.compile('adam', 'categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, epochs=20)
model.evaluate(x_test, to_categorical(y_test))

feature_model = Model(lin, hidden)