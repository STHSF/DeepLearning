# coding=utf-8

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
"""
@function: lstm预测cosin。
说明：最后的训练数据是(seq,res),其中seq = sin(x),res = cos(x),也即该LSTM模型所学习到的是sin(x)到cos(x)的映射关系,
最后给定一个输入sin(x0),LSTM能够预测出相对应的cos(x0).
而不应该理解成输入sin(x0),sin(x1),sin(x2),...,sin(x(n-1)),然后去预测sin(x(n))
@version: 0.1
@author: Li Yu
@license: Apache Licence
@file: data_process.py
@time:
"""

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


class lstm_model(object):
    def __init__(self, num_steps, input_size, output_size, cell_size, batch_size, is_training):
        """
        :param num_steps: 每批数据总包含多少时间刻度(序列长度)
        :param input_size: 输入数据的维度
        :param output_size: 输出数据的维度 如果是类似价格曲线的话，应该为1
        :param cell_size: cell的大小
        :param batch_size: 每批次训练数据的数量
        """
        self.num_steps = num_steps   # 读取长度为num_steps的子序列
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
            self.add_multi_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('loss'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)   # RMSPropOptimizer(0.001, 0.9)

    def add_input_layer(self):
        # initialize input_data, weights, biases
        # l_in_x:(batch*n_step, in_size),相当于把这个批次的样本串到一个长度1000的时间线上，每批次50个样本，每个样本20个时刻
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch_size * num_step, input_size)

        # Ws (in_size, HIDDEN_SIZE)
        weights_in = self._weight_variable([self.input_size, self.HIDDEN_SIZE])

        # bs (HIDDEN_SIZE, )
        biases_in = self._bias_variable([self.HIDDEN_SIZE, ])

        # l_in_y = (batch * time_steps, HIDDEN_SIZE)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, weights_in) + biases_in

        # reshape l_in_y ==> (batch_size, num_steps, HIDDEN_SIZE)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.num_steps, self.HIDDEN_SIZE], name='2_3D')

    def rnn_cell(self):
        # Or GRUCell, LSTMCell(args.hiddenSize)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_SIZE,
                                                 state_is_tuple=True)
        if not self.is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                      input_keep_prob=1.0,
                                                      output_keep_prob=KEEP_PROB)
        return lstm_cell

    def add_multi_cell(self):
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_SIZE, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.MultiRNNCell([self.rnn_cell() for _ in range(NUM_LAYERS)], state_is_tuple=True)

        with tf.name_scope('initial_state'):
            # 初始化最初的状态
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        # time_major=False 表示时间主线不是第一列batch
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                                     inputs=self.l_in_y,
                                                                     initial_state=self.cell_init_state,
                                                                     time_major=False)

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
            self._cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('loss', self.cost)

    def ms_error(self, y_pre, y_target):

        return tf.square(tf.subtract(y_pre, y_target))

    def _weight_variable(self, shape, name='weights_1'):

        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)

        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases_1'):

        initializer = tf.constant_initializer(0.1)

        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    @property
    def cost(self):
        return self._cost


if __name__ == '__main__':

    model = lstm_model(NUM_STEPS, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE, True)
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
            [model.train_op, model.cost, model.cell_final_state, model.pred], feed_dict=feed_dict)

        # plotting
        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:NUM_STEPS], 'b--')
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)

        if i % 20 == 0:
            print('loss: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)