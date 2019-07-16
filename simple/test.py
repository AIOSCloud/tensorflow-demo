# _*_coding:utf-8_*_
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt
from scipy import signal

# image = Image.open('/Users/icsoc/Downloads/28x28.png')
# image_x = np.array(image)
# image_x1 = image_x.reshape([-1, 28, 28, 1])
# image_x2 = image_x.reshape([4, 28, 28, 1])
# weight = np.ones([5, 5, 1, 32])
# conv1 = signal.convolve2d(image_x1, weight, 'same', 'fill',0)
# print(conv1)

# case 1
# 输入是一张3*3大小的图片,图像通道数为5,卷积和大小为1*1,数量为1
# 步长是[1,1,1,1]最后得到一个3*3的feature map
# 1张图最后输出就是一个shape为[1,3,3,1]的张量
input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
print(input)
filter = tf.Variable(tf.random_normal([1, 1, 5, 1]))
op1 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# case 2
# 输入一张3*3大小的图片,图像通道数为5,卷积核是2*2大小,数量是1
# 步长是[1,1,1,1],最后得到一个3*3的feature map
# 1张图最后输出的就是一个shape为[1,3,3,1]的张量
input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
filter = tf.Variable(tf.random_normal([2, 2, 5, 1]))
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# case 3
# 输入一张3*3大小的图片,图像通道数为5,卷积和大小为3*3,数量为1
# 步长是[1,1,1,1]最后得到一个1*1的feature map(不考虑边界)
# 1张图最后输出一个shape为[1,1,1,1]的张量
input = tf.Variable(tf.random_normal([1, 3, 3, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))
op3 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

# case 4
# 输入是一张5*5的大小的图片,图像通道数为5,卷积核是3*3大小,数量是1
# 步长是[1,1,1,1],最后得到一个3*3的feature map (不考虑边界)
# 1张图最后输出一个shape为[1,3,3,1]的张量
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))
op4 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

# case 5
# 输入一张5*5大小的图片,图像通道数为5,卷积核带下为3*3,数量为1
# 步长为[1,1,1,1]最后得到一个5*5的feature map(考虑边界)
# 1 张图最后输出的就是一个shape为[1,5,5,1]的张量
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))
op5 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# case 6
# 输入是1张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是7
# 步长是[1,1,1,1]最后得到一个 5*5 的feature map (考虑边界)
# 1张图最后输出就是一个 shape为[1,5,5,7] 的张量
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))
op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# case 7
# 输入是1张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是7
# 步长是[1,2,2,1]最后得到7个 3*3 的feature map (考虑边界)
# 1张图最后输出就是一个 shape为[1,3,3,7] 的张量
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))
op7 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')

# case 8
# 输入是10 张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是7
# 步长是[1,2,2,1]最后每张图得到7个 3*3 的feature map (考虑边界)
# 10张图最后输出就是一个 shape为[10,3,3,7] 的张量
input = tf.Variable(tf.random_normal([10, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))
op8 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')

# case 9
# 输入是50张28*28大小的图片,图像通道为1,卷积核的大小为5*5
input = tf.Variable(tf.random_normal([50, 28, 28, 1]))
filter = tf.Variable(tf.random_normal([5, 5, 1, 64]))
op9 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print('*' * 20 + ' op1 ' + '*' * 20)
    print(sess.run(op1))
    print('*' * 20 + ' op2 ' + '*' * 20)
    print(sess.run(op2))
    print('*' * 20 + ' op3 ' + '*' * 20)
    print(sess.run(op3))
    print('*' * 20 + ' op4 ' + '*' * 20)
    print(sess.run(op4))
    print('*' * 20 + ' op5 ' + '*' * 20)
    print(sess.run(op5))
    print('*' * 20 + ' op6 ' + '*' * 20)
    print(sess.run(op6))
    print('*' * 20 + ' op7 ' + '*' * 20)
    print(sess.run(op7))
    print('*' * 20 + ' op8 ' + '*' * 20)
    print(sess.run(op8))
    print('*' * 20 + ' op9 ' + '*' * 20)
    print(sess.run(op9))
