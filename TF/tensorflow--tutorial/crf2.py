import tensorflow as tf
import numpy as np

batch_size = 128
sequence_length = 100

batch_size2 = tf.placeholder(tf.int32, [])
# sequence_length2 = tf.placeholder(tf.int32, [])
sequence_length2 = 100

_sequence_length1 = tf.convert_to_tensor(batch_size * [sequence_length], dtype=tf.int32)
# _sequence_length2 = tf.convert_to_tensor(batch_size2 * [sequence_length2], dtype=tf.int32)
# _sequence_length2 = tf.convert_to_tensor(batch_size2 * [sequence_length2], dtype=tf.int32)
# _sequence_length2 = tf.tile([sequence_length2], [batch_size2])
_sequence_length2 = tf.constant(np.full(batch_size, sequence_length2 - 1, dtype=np.int32))


with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print('_sequence_length1', session.run(_sequence_length1))
    print('shape of sequence1', np.shape(session.run(_sequence_length1)))

    print('_sequence_length2', session.run(_sequence_length2, feed_dict={batch_size2: 128}))
    print('shape of sequence2', np.shape(session.run(_sequence_length2, feed_dict={batch_size2: 128})))

