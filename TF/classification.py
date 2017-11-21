# coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def add_layer(inputs, in_size, out_size, activation_function=None):
    """添加层"""
    # add one more layer and return the output of this layer
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases

    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b, )

    return outputs


def compute_accuracy(v_xs, v_ys):

    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})

    return result


'''1.训练的数据'''
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

'''2.定义节点准备接收数据'''
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

'''3.定义神经层：隐藏层和预测层'''
# add hidden layer 输入值是 xs，在隐藏层有 1000 个神经元
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)  # softmax是激励函数的一种

# add output layer 输入值是隐藏层 l1，在预测层输出 10 个结果
# prediction = add_layer(l1, 800, 10, activation_function=None)

'''4.定义 loss 表达式'''
# 分类问题
# the error between prediction and real data
# loss 函数用 cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entropy)

# 初始化session
sess = tf.Session()

# 参数初始化
sess.run(tf.initialize_all_variables())

# 迭代优化
for i in range(2000):

    batch_xs, batch_ys = mnist_data.train.next_batch(100)
    print "batch_xs.shape"
    print batch_xs.shape
    print "batch_ys.shape"
    print batch_ys.shape


    sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys})

    if i % 10 == 0:
        print(compute_accuracy(mnist_data.test.images, mnist_data.test.labels))







