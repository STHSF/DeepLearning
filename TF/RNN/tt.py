#!/usr/bin/env python
# coding=utf-8

"""
@function:
@version: ??
@author: Li Yu
@license: Apache Licence 
@file: test.py
@time: 2017/6/16 上午9:58
"""
import numpy as np
BATCH_START = 0
NUM_STEPS = 20  # 训练时每个序列的长度，理论上lstm可以处理任意序列的长度，但是为了避免梯度消失的问题，回国定一个最大的序列长度。
BATCH_SIZE = 50  # 优化lstm时，每次会使用一个batch的训练样本。
INPUT_SIZE = 1
OUTPUT_SIZE = 1
HIDDEN_SIZE = 30  # hidden_unit_size
LR = 0.006
KEEP_PROB = 0.5
NUM_LAYERS = 10


def get_batch():
    """
    训练样本生成函数，根据batch_size和num_steps生成训练样本
    :return:
    """
    global BATCH_START, NUM_STEPS
    # xs shape (50 batch_size, 20 num_steps)
    xs = np.arange(BATCH_START, BATCH_START + NUM_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, NUM_STEPS)) / (10 * np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += NUM_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    # np.newaxis 用来增加一个维度 变为三个维度，第三个维度将用来存上一批样本的状态
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

[seq, res, xs] = get_batch()

print('seq.shape',np.shape(seq))
print('res.shape',np.shape(res))
print('xs.shape',np.shape(xs))

# print('res\n', res)
# print('xs\n', xs)