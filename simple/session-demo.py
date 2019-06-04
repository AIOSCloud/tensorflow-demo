# _*_coding:utf-8_*_
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 数据加载
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 初始化session
sess = tf.InteractiveSession()

# 出事化变量 占位符
# 图片像素展开
x = tf.placeholder("float", shape=[None, 784])
# 预测结果值
y_ = tf.placeholder("float", shape=[None, 10])

# 变量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 训练模型
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 循环计算
for i in range(1000000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    pass

# 评估模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
