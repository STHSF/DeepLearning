#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: ??
@author: Li Yu
@license: Apache Licence 
@file: keras_test.py
@time: 2017/6/15 下午4:08
"""

import os
import time
import warnings
import numpy as np
from numpy import newaxis
import tensorflow as tf
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
# model.add(LSTM(32, activation='relu', input_shape=(1, 100)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
data = np.random.random((1000, 100, 1))
data1 = np.random.random((1000, 1, 100))
labels = np.random.randint(2, size=(1000, 1))
print(data)
print(data1)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)