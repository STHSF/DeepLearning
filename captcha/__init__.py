# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import glob
import random
from PIL import Image
from skimage import io
import configparser
import os

# config=configparser.ConfigParser()
# config.read(os.getcwd()+"\\conf.cfg")
# PATH=config.get('input','inputDir')
# MODEL_PATH=config.get('model','modelpath')

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 100
MAX_CAPTCHA = 5
CHAR_SET_LEN = 63
PATH = "D:\\tmp\\picture_test\\*.png"


def getPicture(path):
    return glob.glob(path)


def getSplitData(path):
    result = getPicture(path)
    length = len(result)
    trainLengh = int(length * 0.9)
    train = result[0:trainLengh]
    test = result[trainLengh:length - 1]
    # train = result[0:int(length * 0.8)]
    # test = [i for i in result if i not in train]
    return train, test


def sampleTrain(length, trainData):
    return random.sample(trainData, length)


# 把彩色图像转化为灰度图像
def convert2gray(image):
    if len(image.shape) > 2:
        grap = np.mean(image, -1)
        return grap
    else:
        return image


# 文本转向量
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError("验证码最长是5个字符")
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


# 向量转文本
def vec2text(vec):
    if not isinstance(vec, list):
        char_pos = vec.nonzero()[0]
    else:
        char_pos = vec
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        text.append(chr(char_code))
    return "".join(text)


def getImageAndName(path):
    name = path.split("\\")[-1].split(".")[0]
    # captcha_image = Image.open(path)
    # captcha_image = np.array(captcha_image)
    img = 1.0 - io.imread(path, as_grey=True)
    return name, img


def get_next_batch(data):
    batch_size = len(data)
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    for i in range(batch_size):
        text, image = getImageAndName(data[i])
        # image = convert2gray(image)
        # batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_x[i, :] = image.flatten()
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y


####################################################################

# 定义CNN
def crack_captcha_cnn(X, keep_prob, w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    # print(x.get_shape())

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # print(conv1.get_shape())

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # print(conv2.get_shape())

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    # print(conv3.get_shape())

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([5 * 13 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


traindata, testdata = getSplitData(PATH)


# 训练
def train_crack_captcha_cnn(max_step=200):
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
    keep_prob = tf.placeholder(tf.float32)  # dropout
    output = crack_captcha_cnn(X, keep_prob)
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output, Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            batch_x, batch_y = get_next_batch(sampleTrain(100, traindata))
            _, lossSize = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            if step % 5 == 0:
                print("step is:" + str(step), u"损失函数大小为" + str(lossSize))
                batch_x_test, batch_y_test = get_next_batch(testdata)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print("step is:" + str(step), "acc is :" + str(acc))
                if step == max_step:
                    saver.save(sess, "./model/crack_capcha.model")
                    break
            step += 1


# batch_x_test, batch_y_test = get_next_batch(testdata)
#				acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
#				print("step is:"+str(step),"acc is :"+str( acc))
#				# 如果准确率大于50%,保存模型,完成训练

def crack_captcha(captcha_image):
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    # Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
    keep_prob = tf.placeholder(tf.float32)
    output = crack_captcha_cnn(X, keep_prob)
    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph('crack_capcha.model.meta')
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        text = text_list[0].tolist()
        return text


def predict(testdata):
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    keep_prob = tf.placeholder(tf.float32)
    output = crack_captcha_cnn(X, keep_prob)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        batch_size = len(testdata)
        count = 0
        for i in range(batch_size):
            text, image = getImageAndName(testdata[i])
            # image = convert2gray(image)
            # captcha_image = image.flatten() / 255
            captcha_image = image.flatten()
            predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
            predict_text = text_list[0].tolist()
            predict_value = vec2text(predict_text)
            flag = text == predict_value
            if flag:
                count += 1
            print("真实值: {}, 预测值: {}, 是否相等: {}".format(text, predict_value, flag))
        print('\n识别结果: {}/{}={}'.format(count, batch_size, count / batch_size))


def predict_single(image_file):
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    keep_prob = tf.placeholder(tf.float32)
    output = crack_captcha_cnn(X, keep_prob)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        text, image = getImageAndName(image_file)
        captcha_image = image.flatten()
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        predict_text = text_list[0].tolist()
        predict_value = vec2text(predict_text)
        print('\n识别结果: {}'.format(predict_value))


if __name__ == '__main__':
    train_crack_captcha_cnn(max_step=5000)
