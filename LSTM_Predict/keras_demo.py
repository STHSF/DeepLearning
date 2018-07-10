#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: ??
@author: Li Yu
@license: Apache Licence 
@file: t1.py
@time: 2017/6/26 上午10:29
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

x_train = np.random.randn(1000, 100)
y_labels = np.random.randint(0, 10, size=(1000,))
y_train = np.zeros((1000, 10))
y_train[range(1000), y_labels] = 1

# print(np.shape(y_labels))
# print(y_labels)


model.fit(x_train, y_train, epochs=5, batch_size=20)

x_test = np.random.randn(1000, 100)
y_labels = np.random.randint(0, 10, size=(1000,))
y_test = np.zeros((1000, 10))
y_test[range(1000), y_labels] = 1

classes = model.predict_classes(x_test, batch_size=20)
proba = model.predict_proba(x_test, batch_size=20)

print(classes)
print(proba)