#!/usr/bin/python3
import numpy as np
from datetime import datetime
import tensorflow as tf
import math

# http://tanzimsaqib.com/redir.html?https://raw.githubusercontent.com/tsaqib/ml-playground/master/cnn-tensorflow/cnn-tensorflow.html

class classifierCNN:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=(None, (32, 32, 3)))
        self.y = tf.placeholder(tf.float32, shape=(None, 10))

		self.keep_prob = tf.placeholder(tf.float32)

		conv1 = conv_layer(self.x, shape=[5,5,3,32])
		conv1_pool = max_pool_2x2(conv1)

		conv2 = conv_layer(conv1_pool, shape=[5,5,32,64])
		conv2_pool = max_pool_2x2(conv2)
		conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])

		full1 = tf.nn.relu(full_layer(conv2_flat, 1024))
		full1_drop = tf.nn.dropout(full_1, keep_prob=self.keep_prob)

		self.output = full_layer(full1_drop,10)

		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output, self.y))
		update = tf.train.AdamOptimizer(self.learningRate).minimize(loss)

		predict = tf.softmax(self.output, dim=1)
