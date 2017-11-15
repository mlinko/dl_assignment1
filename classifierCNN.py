#!/usr/bin/python3
import numpy as np
from datetime import datetime
import tensorflow as tf
import math

# http://tanzimsaqib.com/redir.html?https://raw.githubusercontent.com/tsaqib/ml-playground/master/cnn-tensorflow/cnn-tensorflow.html

class classifierCNN:
	def __init__(self):
		self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
		self.y = tf.placeholder(tf.float32, shape=(None, 10))

		self.keep_prob = tf.placeholder(tf.float32)

		conv1 = conv_layer(self.x, shape=[5,5,3,32])
		conv1_pool = max_pool_2x2(conv1)
		self.learningRate = 0.001

		conv2 = conv_layer(conv1_pool, shape=[5,5,32,64])
		conv2_pool = max_pool_2x2(conv2)
		conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])

		full1 = tf.nn.relu(full_layer(conv2_flat, 1024))
		full1_drop = tf.nn.dropout(full1, keep_prob=self.keep_prob)

		self.output = full_layer(full1_drop,10)

	def train(self, Xin, Yin, Xtein, Yte, epochs=1, batchSize=100):
		X = Xin.reshape(Xin.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
		Xte = Xtein.reshape(Xtein.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
		Y = np.zeros([len(Yin), 10], dtype=np.float32)
		for i in range(len(Y)):
			Y[i, Yin[i]] = 1

		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.output, labels=self.y))
		update = tf.train.AdamOptimizer(self.learningRate).minimize(loss)

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		with tf.Session() as session:
			session.run(init)
			for epoch in range(epochs):
				for i in range(math.ceil(X.shape[0]/batchSize)):
					batchStart = i * batchSize
					batchEnd = batchStart + batchSize
					if batchEnd > len(X): batchEnd = len(X)
					if i%10 == 0:
						loss = session.run(loss, feed_dict={self.x: X[batchStart:batchEnd, :],
						                                    self.y: Y[batchStart:batchEnd]})
						print('[%s]: epoch %3d, batch %3d, loss %f' % (datetime.now().strftime('%Y.%m.%d %H:%M:%S'), epoch, i, loss))

					c = session.run(updates, feed_dict={self.x: X[batchStart:batchEnd,:], self.y: Y[batchStart:batchEnd]})
				for layer in tf.trainable_variables():
					if 'w' in layer.name:
						print(layer.name)
						print(session.run(layer))
				guess = np.argmax( session.run(self.predictor, feed_dict={self.x: Xte }), axis=1)
				accuracy = np.mean(Yte == guess)
				print('[%s]: epoch %3d, accuracy %2.1f %%'%(datetime.now().strftime('%Y.%m.%d %H:%M:%S'), epoch, accuracy *100))
				savePath = saver.save(session,'cnn/1_epoch%d/model.ckpt'%epoch)
				X, Y = shuffleData(X,Y)

	def predict(self):
		predict = tf.nn.softmax(self.output, dim=1)


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
	W = weight_variable(shape)
	b = bias_variable([shape[3]])
	return tf.nn.relu(conv2d(input, W)+ b)

def full_layer(input, size):
	input_size = int(input.get_shape()[1])
	W = weight_variable([input_size, size])
	b = bias_variable([size])
	return tf.matmul(input,W) + b 

def shuffleData(x,y):
	assert len(x) == len(y)
	indexes = np.arange(len(x))
	np.random.shuffle(indexes)
	xout = x.copy()
	yout = y.copy()
	for i in indexes:
		xout[i] = x[ indexes[i]]
		yout[i] = y[ indexes[i]]
	return xout, yout

if __name__ == '__main__':
	from cifar10reader import loadInputs
	Xtr, Ytr, Xte, Yte, dictionary = loadInputs('cifar-10-batches-py')
	cnn = classifierCNN()
	cnn.train(Xtr, Ytr, Xte, Yte)
