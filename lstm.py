# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 19:11:02 2022

@author: halid
"""

import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Input, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np

data = pd.read_csv("authors.csv")
data['name'] = data['name'].apply(str.lower)

vocabulary_list = list(set(' '.join(data['name'])))
vocabulary = dict(zip(vocabulary_list, range(len(vocabulary_list))))
NCHAR = len(vocabulary)

data_matrix = [[vocabulary.get(c) for c in row] for row in data['name']]
data_matrix = pad_sequences(data_matrix)
MAXLEN = data_matrix.shape[1]

x = data_matrix[:,0:35]
y = data_matrix[:,35]


lin = Input(35)
embedding = Embedding(NCHAR, 80)(lin)
lstm = LSTM(80, return_sequences=True)(embedding)
lstm = LSTM(80)(lstm)
out = Dense(NCHAR, activation='softmax')(lstm)

model = Model(lin, out)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

model.fit(x, to_categorical(y), epochs=100)

def test(name:str):
    ids = [[vocabulary.get(c) for c in name.lower()]]
    y_pred = model.predict(pad_sequences(ids, maxlen=MAXLEN))[0]
    return vocabulary_list[np.argmax(y_pred)]
    


