# coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 添加层,带有激活函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    """添加层"""
    # add one more layer and return the output of this layer
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases

    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs


'''1.训练的数据'''
# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

'''2.定义节点准备接收数据'''
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32, [None, 1])

'''3.定义神经层：隐藏层和预测层'''
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
# 输入层只有一个属性,inputs=1
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)  # relu是激励函数的一种
l2 = add_layer(l1, 10, 20, activation_function=tf.nn.relu)
l2 = add_layer(l1, 20, 30, activation_function=tf.nn.relu)

# 输出层也只有一个属性,outputsize=1
# add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
p1 = add_layer(l2, 30, 10, activation_function=tf.nn.relu6)
prediction = add_layer(p1, 10, 1, activation_function=None)

'''4.定义 loss 表达式'''
# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))  # 需要向相加索引号,reduce执行跨纬度操作

'''5.选择 optimizer 使 loss 达到最小'''
# 这一行定义了用什么方式去减少 loss，学习率是 0.1
optimizer = tf.train.RMSPropOptimizer(0.0001)
train = optimizer.minimize(loss)  # 选择梯度下降法

# important step 对所有变量进行初始化
init = tf.initialize_all_variables()
sess = tf.Session()

# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_data, y_data, "-")
plt.ion()
plt.show()

for i in range(2000):
    # training
    sess.run(train, feed_dict={xs: x_data, ys: y_data})

    if i % 10 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=2)
        plt.pause(0.1)
