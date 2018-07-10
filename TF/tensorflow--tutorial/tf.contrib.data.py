# coding=utf-8

# tensorflow tf.contrib.data api test


import tensorflow as tf


# file path
filename = ''
batch_size = 100
aa = (tf.contrib.data.TextLineDataset(filename)
      .map((lambda line: tf.decode_csv(line, record_defaults=[['1'], ['1'], ['1']], field_delim='\t')))
      .shuffle(buffer_size=1000)
      .batch_size(batch_size)
      )


