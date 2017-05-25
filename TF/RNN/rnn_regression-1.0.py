# coding=utf-8

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
"""
@function: lstm预测cosin。
@version: 0.1
@author: Li Yu
@license: Apache Licence
@file: data_process.py
@time:
"""

BATCH_START = 0
NUM_STEPS = 20
BATCH_SIZE = 50  # 每个batch的大小
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006
KEEP_PROB = 0.5
NUM_LAYERS = 5


def get_batch():
    # 生成数据
    global BATCH_START, NUM_STEPS
    # xs shape (50 batch_size, 20 time_steps)
    xs = np.arange(BATCH_START, BATCH_START + NUM_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, NUM_STEPS)) / (10 * np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += NUM_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


class lstm_model(object):
    def __init__(self, num_steps, input_size, output_size, cell_size, batch_size, is_training):
        self.num_steps = num_steps   # 读取长度为n_steps的子序列
        self.input_size = input_size
        self.output_size = output_size
        self.HIDDEN_SIZE = cell_size
        self.batch_size = batch_size
        self.is_training = is_training
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, num_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, num_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)   # RMSPropOptimizer(0.001, 0.9)

    def add_input_layer(self):
        # initialize input_data, weights, biases

        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)

        # Ws (in_size, HIDDEN_SIZE)
        weights_in = self._weight_variable([self.input_size, self.HIDDEN_SIZE])

        # bs (HIDDEN_SIZE, )
        biases_in = self._bias_variable([self.HIDDEN_SIZE, ])

        # l_in_y = (batch * time_steps, HIDDEN_SIZE)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, weights_in) + biases_in

        # reshape l_in_y ==> (batch, time_steps, HIDDEN_SIZE)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.num_steps, self.HIDDEN_SIZE], name='2_3D')

    def rnn_cell(self):
        # Or GRUCell, LSTMCell(args.hiddenSize)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_SIZE, state_is_tuple=True)
        if not self.is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=1.0,
                                                         output_keep_prob=KEEP_PROB)
        return lstm_cell

    def add_cell(self):
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_SIZE, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.MultiRNNCell([self.rnn_cell() for _ in range(NUM_LAYERS)], state_is_tuple=True)

        with tf.name_scope('initial_state'):
            # 初始化最初的状态
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):

        # shape = (batch * steps, HIDDEN_SIZE)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.HIDDEN_SIZE], name='2_2D')

        weights_out = self._weight_variable([self.HIDDEN_SIZE, self.output_size])

        biases_out = self._bias_variable([self.output_size, ])

        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, weights_out) + biases_out

    def compute_cost(self):

        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.num_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )

        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    def ms_error(self, y_pre, y_target):

        return tf.square(tf.subtract(y_pre, y_target))

    def _weight_variable(self, shape, name='weights_1'):

        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)

        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases_1'):

        initializer = tf.constant_initializer(0.1)

        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':

    model = lstm_model(NUM_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, True)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    sess.run(tf.global_variables_initializer())
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    plt.ion()
    plt.show()
    for i in range(500):
        seq, res, xs = get_batch()
        if i == 0:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                # create initial state
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_init_state: state    # use last state as the initial state for this run
            }

        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)

        # plotting
        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:NUM_STEPS], 'b--')
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)

        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)