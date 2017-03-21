# coding=utf-8
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


# read_data
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    """
    import data_sets
    """
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save
    print "pre_train_dataset", train_dataset.shape, train_labels.shape
    print "pre_valid_dataset", valid_dataset.shape, valid_labels.shape
    print "pre_test_dataset", test_dataset.shape, test_labels.shape
# image size（28 * 28）
image_size = 28
# the numbers of labels
num_labels = 10


def data_format(input_data, labels):
    """
    reshape data_sets. convert the matrix to a vector, [(28, 28)] ==> [(1, 28*28)]
    reshape labels. convert the label to one-hot code,  0 ==> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    :param input_data:
    :param labels:
    :return:
    """
    data = input_data.reshape([-1, image_size * image_size]).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return data, labels


train_dataset, train_labels = data_format(train_dataset, train_labels)
valid_dataset, valid_labels = data_format(valid_dataset, valid_labels)
test_dataset, test_labels = data_format(test_dataset, test_labels)
print "train_dataset", train_dataset.shape, train_labels.shape
print "valid_dataset", valid_dataset.shape, valid_labels.shape
print "test_dataset", test_dataset.shape, test_labels.shape

# setting hidden units
hidden_units = 1024
# setting batch size
batch_size = 128

graph = tf.Graph()
with graph.as_default():
    # we designed two layers for this neural network, so we have two weights and biases
    with tf.name_scope('weight'):
        weight = {
            # the weights of first layer
            "w1": tf.Variable(tf.truncated_normal([image_size * image_size, hidden_units]), name="weight1"),
            # the weights of second layer
            "w2": tf.Variable(tf.truncated_normal([hidden_units, num_labels]), name="weight2")
        }

    with tf.name_scope('biases'):
        biases = {
            # the biases of first layer
            "b1": tf.Variable(tf.zeros([hidden_units]), name="biases1"),
            # the biases of second layer
            "b2": tf.Variable(tf.zeros([num_labels]), name="biases2")
        }


    def multi_layers(input_data, weight, baies):
        """

        :param input_data:
        :param weight:
        :param baies:
        :return:
        """
        with tf.name_scope('layer_1'):
            logits_1 = tf.matmul(input_data, weight['w1']) + baies['b1']
        with tf.name_scope('relu'):
            hidden_layer = tf.nn.relu(logits_1, name='hidden_layer')
        with tf.name_scope('layer_2'):
            logits_2 = tf.matmul(hidden_layer, weight['w2']) + baies['b2']

        return logits_2


    with tf.name_scope('input_data'):
        # create placeholder for train, valid, test datasets
        with tf.name_scope('train_data'):
            tf_train_data = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        with tf.name_scope('train_labels'):
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        with tf.name_scope('valid_data'):
            tf_valid_data = tf.constant(valid_dataset)
        with tf.name_scope('test_data'):
            tf_test_data = tf.constant(test_dataset)

    with tf.name_scope('loss'):
        predict = multi_layers(tf_train_data, weight, biases)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=predict, name='loss'))

    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    with tf.name_scope('train_prediction'):
        train_prediction = tf.nn.softmax(predict)
    with tf.name_scope('valid_prediction'):
        valid_predict = multi_layers(valid_dataset, weight, biases)
        valid_prediction = tf.nn.softmax(valid_predict)
    with tf.name_scope('test_prediction'):
        test_predict = multi_layers(test_dataset, weight, biases)
        test_prediction = tf.nn.softmax(test_predict)

num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    # use command "tensorboard --logdir='/logss'" to show the tensor graph
    writer = tf.summary.FileWriter("logss/", session.graph)
    saver = tf.train.Saver()

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
        feed_dict = {tf_train_data: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

    # 模型保存
    # saver.save(session, '/model')
