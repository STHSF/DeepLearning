#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: ??
@author: Li Yu
@license: Apache Licence 
@file: t.py
@time: 2017/6/1 上午10:09
"""

import tensorflow as tf
import numpy as np


tf.reset_default_graph()

# Create input data
X = np.random.randn(2, 10, 8)

print('X.shape:', np.shape(X))

# 第二个example的长度为6
X[1, 6:] = 0
X_lengths = [10, 6]

# 隐藏层的单元个数是64
cell = tf.contrib.rnn.BasicLSTMCell(num_units=64, state_is_tuple=True)


outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    inputs=X,
    dtype=tf.float64,
    sequence_length=X_lengths)

result = tf.contrib.learn.run_n(
    {"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)

assert result[0]["outputs"].shape == (2, 10, 64)
print(result[0]["outputs"])

# Outputs for the second example past past length 6 should be 0
# 第二个example中的outputs超过6步(7-10步)的值应该为0
assert (result[0]["outputs"][1, 7, :] == np.zeros(cell.output_size)).all()