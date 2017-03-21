import tensorflow as tf


def conv_relu(inputs, kenal_shape, biases_shape):
    weights = tf.get_variable("weights_1", kenal_shape, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases_1", biases_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')

    return tf.nn.relu(conv + biases)


def image_filter(inputs_images):
    with tf.variable_scope("conv1"):
        conv1 = conv_relu(inputs_images, [5, 5, 32, 32], [32])

    with tf.variable_scope("conv2"):
        conv2 = conv_relu(conv1, [5, 5, 32, 32], [32])

        return conv2


with tf.variable_scope("image_filter") as scope:
    res1 = image_filter(image1)
    scope.reuse_variable()
    res2 = image_filter(image2)

