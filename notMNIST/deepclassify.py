import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


pickle_file = "notMNIST.pickle"
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 128
hidden_units = 1024

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    with tf.name_scope('input'):
        with tf.name_scope('tf_train_dataset'):
            tf_train_dataset = tf.placeholder(tf.float32,
                                              shape=(batch_size, image_size * image_size))
        with tf.name_scope('tf_train_labels'):
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        with tf.name_scope('tf_valid_dataset'):
            tf_valid_dataset = tf.constant(valid_dataset)
        with tf.name_scope('tf_test_dataset'):
            tf_test_dataset = tf.constant(test_dataset)

    # neural_network
    with tf.name_scope('weights'):
        weights_1 = tf.Variable(
            tf.truncated_normal([image_size * image_size, hidden_units]))
    with tf.name_scope('biases_1'):
        biases_1 = tf.Variable(tf.zeros([hidden_units]))
    # Training computation.

    with tf.name_scope('weights_2'):
        weights_2 = tf.Variable(tf.truncated_normal([hidden_units, num_labels]))
    with tf.name_scope('biases_2'):
        biases_2 = tf.Variable(tf.zeros([num_labels]))
    with tf.name_scope('hidden_logits'):
        hidden_logits = tf.matmul(tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1), weights_2) + biases_2

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=hidden_logits))

    # Optimizer.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    with tf.name_scope('train_prediction'):
        train_prediction = tf.nn.softmax(hidden_logits)
    with tf.name_scope('valid_prediction'):
        valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1), weights_2) + biases_2)
    with tf.name_scope('test_prediction'):
        test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2)

num_steps = 3001

with tf.Session() as session:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter("log/", session.graph)
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        #  Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
