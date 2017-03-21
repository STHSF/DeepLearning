#!/usr/bin/env/python
# coding=utf-8
import tensorflow as tf
import numpy as np


input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])

embeding = tf.Variable(np.identity(5, dtype=np.int32))
input_embeding = tf.nn.embedding_lookup(embeding, input_ids)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(embeding.eval())
print(sess.run(input_embeding, feed_dict={input_ids:[[1, 2], [2, 3], [3, 3]]}))
