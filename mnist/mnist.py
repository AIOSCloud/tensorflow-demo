# _*_coding:utf-8_*_
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 初始化变量 y=WX+b
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 神经网络解决多分类问题最常用的方法是设置n个输出节点，其中n为类别的个数。对于每一个样例，神经网络可以得到的一个n维数组作为输出结果。数组中的每一个维度（也就是每一个输出节点）对应一个类别。在理想情况下，如果一个样本属于类别k，那么这个类别所对应的输出节点的输出值应该为1，而其他节点的输出都为0
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 预测值
y_ = tf.placeholder("float", [None, 10])
# 交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 训练步骤
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化/启动图计算
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 训练10000次
for i in range(1000000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 验证结果正确性
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 打印结果
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
