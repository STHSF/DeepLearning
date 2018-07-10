# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
import shutil
# 清空log文件
filelist = []
rootdir = "./model/tensorflowlogs/"
for f in os.listdir(rootdir):
    filepath = os.path.join(rootdir, f)
    if os.path.isfile(filepath):
        os.remove(filepath)
        print filepath+" removed!"
    elif os.path.isdir(filepath):
        shutil.rmtree(filepath, True)
        print "dir "+filepath+" removed!"

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


class model:
    def __init__(self, name_scope, configure, is_training):
        self.batch_size = configure.batch_size
        self.n_steps = configure.n_steps
        self.n_inputs = configure.n_inputs
        self.n_classes = configure.n_classes
        self.n_hidden_units = configure.n_hidden_units
        self.is_training = is_training
        self.lr = configure.lr
        self.graph()
        self.merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, name_scope))

    def graph(self):
        # tf Graph input
        with tf.variable_scope("input_data") as scope:
            self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        with tf.variable_scope("labels") as scope:
            self.y = tf.placeholder(tf.float32, [None, self.n_classes])

        # # Define weights_1
        with tf.variable_scope("weight") as scope:
            self.weights = {
                # (28, 128)
                'in': tf.get_variable('in', initializer=tf.random_normal([self.n_inputs, self.n_hidden_units])),
                # (128, 10)
                'out': tf.get_variable('out', initializer=tf.random_normal([self.n_hidden_units, self.n_classes]))
            }
        with tf.variable_scope("biases"):
            self.biases = {
                # (128, )
                'in': tf.get_variable('in', initializer=tf.constant(0.1, shape=[self.n_hidden_units, ])),
                # (10, )
                'out': tf.get_variable('out', initializer=tf.constant(0.1, shape=[self.n_classes, ]))
            }
        with tf.variable_scope("pre"):
            self.logits = self.rnn(self.x, self.weights, self.biases)
        with tf.variable_scope('loss'):
            _loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        with tf.variable_scope("cost") as scope:
            self.cost = tf.reduce_mean(_loss)
            tf.summary.scalar(scope.name, self.cost)

        with tf.variable_scope("accuracy") as scope:
            correct_predict = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
            tf.summary.scalar(scope.name, self.accuracy)

        if not self.is_training:
            return
        with tf.name_scope("train_op"):
            self.train = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    def rnn(self, input_data, weights, biases):
        # hidden layer for input to cell
        ########################################

        # transpose the inputs shape from
        # X（128 batch ,28 steps, 28 inputs）
        # ==> (128 batch * 28 steps, 28 inputs)
        input_data = tf.reshape(input_data, [-1, self.n_inputs])

        # into hidden
        # data_in = (128 batch * 28 steps, 128 hidden)
        data_in = tf.matmul(input_data, weights['in']) + biases['in']
        # data_in ==> (128 batch, 28 steps, 128 hidden_units)
        data_in = tf.reshape(data_in, [-1, self.n_steps, self.n_hidden_units])

        # cell
        ##########################################
        # basic LSTM Cell.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        # lstm cell is divided into two parts (c_state, h_state)
        _init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, data_in, initial_state=_init_state, time_major=False)

        # hidden layer for output as the final results
        #############################################
        # results = tf.matmul(final_state[1], weights_1['out']) + biases_1['out']

        # # or
        # unpack to list [(batch, outputs)..] * steps
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
        logits = tf.matmul(outputs[-1], weights['out']) + biases['out']

        return logits


class conf(object):
    init_scale = 0.04

    # hyperparameters
    lr = 0.001
    training_iters = 100000
    batch_size = 128
    n_inputs = 28  # MNIST data input (img shape: 28*28)
    n_steps = 28  # time steps
    n_hidden_units = 128  # neurons in hidden layer
    n_classes = 10  # MNIST classes (0-9 digits)


train_conf = conf()
valid_conf = conf()
valid_conf.batch_size = 20
valid_conf.training_iters = 2000
valid_conf.num_steps = 1


def run_epoch(_model, session, _conf):
    """Testing or valid."""
    count = 0
    while count * _conf.batch_size < _conf.training_iters:
        test_batch_xs, test_batch_ys = mnist.test.next_batch(_conf.batch_size)
        test_batch_xs = test_batch_xs.reshape([_conf.batch_size, _conf.n_steps, _conf.n_inputs])
        # print(batch_xs.shape)
        # print(batch_ys.shape)
        session.run([_model.cost, _model.accuracy], feed_dict={_model.x: test_batch_xs, _model.y: test_batch_ys, })
        if count % 3 == 0:
            _summary = session.run(_model.merged, feed_dict={_model.x: test_batch_xs, _model.y: test_batch_ys, })
            test_summary_writer.add_summary(_summary, count)
            print(session.run(_model.accuracy, feed_dict={_model.x: test_batch_xs, _model.y: test_batch_ys, }))
        count += 1


with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-train_conf.init_scale, train_conf.init_scale)

    with tf.name_scope("Train") as train_scope:
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_model = model(train_scope, train_conf, is_training=True)

    with tf.name_scope("Test") as test_scope:
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            test_model = model(test_scope, valid_conf, is_training=False)

    with tf.Session() as session:
        train_summary_writer = tf.summary.FileWriter('./model/tensorflowlogs/train', session.graph)
        test_summary_writer = tf.summary.FileWriter('./model/tensorflowlogs/test')
        session.run(tf.global_variables_initializer())
        step = 0
        while step * train_conf.batch_size < train_conf.training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(train_conf.batch_size)
            batch_xs = batch_xs.reshape([train_conf.batch_size, train_conf.n_steps, train_conf.n_inputs])
            # print(batch_xs.shape)
            # print(batch_ys.shape)
            session.run([train_model.train], feed_dict={train_model.x: batch_xs, train_model.y: batch_ys, })
            if step % 2 == 0:
                summary = session.run(train_model.merged, feed_dict={train_model.x: batch_xs, train_model.y: batch_ys, })
                train_summary_writer.add_summary(summary, step)
                print(session.run(train_model.accuracy, feed_dict={train_model.x: batch_xs, train_model.y: batch_ys, }))
            step += 1
            # print step
            # if step % 1000 == 0 or (step + 1) == train_conf.training_iters:
                # test_batch_xs, test_batch_ys = mnist.test.next_batch(valid_conf.batch_size)
                # test_batch_xs = test_batch_xs.reshape([valid_conf.batch_size, valid_conf.n_steps, valid_conf.n_inputs])
                # print(session.run(test_model.accuracy, feed_dict={test_model.x: test_batch_xs, test_model.y: test_batch_ys, }))
        run_epoch(test_model, session, valid_conf)

