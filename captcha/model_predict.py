#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: model_predict.py
@time: 2018/7/9 下午8:33
"""
import os
import numpy as np
import tensorflow as tf
from gen_captcha import gen_captcha_text_and_image, number, alphabet, ALPHABET
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽log信息、INFO信息和WARNING信息

print('=========================获取验证码=============================')
text, _image = gen_captcha_text_and_image()  # 先检验生成验证码和文字测试模块是否完全
print("验证码图像channel: {}".format(_image.shape))  # (50, 200, 3)
# 图像大小
IMAGE_HEIGHT = 50
IMAGE_WIDTH = 200
MAX_CAPTCHA = len(text)
# MAX_CAPTCHA = 4
print("验证码文本最长字符数: %s" % MAX_CAPTCHA)  # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐
print('=========================验证码获取结束==========================')


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


# 文本转向量
char_set = number + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)
print('CHAR_SET_LEN: {}'.format(CHAR_SET_LEN))


def text2vec(text):
    # 总共52个字母加10个数字，还有一个下划线，总共63个字符
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')
    # vector: (252,)
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(_char):
        """
        将字符转化成序号
        :param _char:
        :return:
        """
        if _char == '_':
            k = 10   # 只有数字的情况下，下划线排在第十一个
            return k
        # 字符编码
        # 如果c为数字
        k = ord(_char) - 48
        if k > 10 or k < 0:  # 如果c为非数字和下划线
            raise ValueError('No Map')
        print('k: %s' % k)
        return k
    # 遍历验证码的每一个字符
    for i, c in enumerate(text):
        print(i, c)
        idx = i * CHAR_SET_LEN + char2pos(c)
        print('idx: %s'% idx)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        # elif char_idx < 36:
        #     char_code = char_idx - 10 + ord('A')
        # elif char_idx < 62:
        #     char_code = char_idx - 36 + ord('a')
        elif char_idx == 10:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


# 向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每11个编码一个字符，这样数字有，字符也有

# vec = text2vec("352_")
# print(np.shape(vec))
# print(vec)
# text = vec2text(vec)
# print(text)  # 352_


def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    def wrap_gen_captcha_text_and_image():
        while True:
            text, _image = gen_captcha_text_and_image()
            if _image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
                return text, _image

    for i in range(batch_size):
        text, _image = wrap_gen_captcha_text_and_image()
        image = convert2gray(_image)

        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


####################################################################
# 申请占位符 按照图片
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])  # (?, 50, 200, 1)
    print('shape_of_x: %s' % x.get_shape())

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer # 3 个 转换层
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))  # (?, 50, 200, 32)
    # print('shape_of_conv1: %s' % conv1.get_shape())
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)  # (?, 25, 100, 32)
    # print('shape_of_conv1_poll_drop: %s' % conv1.get_shape())

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))  # (?, 25, 100, 64)
    # print('shape_of_conv2: %s' % conv2.get_shape())
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)  # (?, 13, 50, 64)
    # print('shape_of_conv2_poll_drop: %s' % conv2.get_shape())

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))  # (?, 13, 50, 64)
    # print('shape_of_conv3: %s' % conv3.get_shape())
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)  # (?, 7, 25, 64)
    # print('shape_of_conv3_poll_drop: %s' % conv3.get_shape())

    # Fully connected layer  # 最后连接层
    w_d = tf.Variable(w_alpha * tf.random_normal([7 * 25 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])  # (?, 11200)
    # print('shape_of_dense: %s' % dense.get_shape())
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)
    # print('shape_of_dense_poll_drop: %s' % dense.get_shape())

    # 输出层
    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out) # (?, 44)
    # print('shape_of_out: %s' % out.get_shape())

    # out = tf.nn.softmax(out)
    return out


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./crack_capcha4/'))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * CHAR_SET_LEN + n] = 1
            i += 1
        return vec2text(vector)


if __name__ == '__main__':
    text, image = gen_captcha_text_and_image()
    image = convert2gray(image)  # 生成一张新图
    image = image.flatten() / 255  # 将图片一维化
    predict_text = crack_captcha(image)  # 导入模型识别
    print("正确: {}  预测: {}".format(text, predict_text))
