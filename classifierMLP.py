#!/usr/bin/python3
import numpy as np
from datetime import datetime
import tensorflow as tf
import math

# build this network with the help of the following:
# - https://www.slideshare.net/DaleiLEE/applying-neural-networks-to-cifar10-dataset
# - https://gist.github.com/vinhkhuc/e53a70f9e5c3f55852b0#file-simple_mlp_tensorflow-py-L8
# - https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py 

class classifierMLP:
	def __init__(self):
		print('[%s]: %s'%(datetime.now().strftime('%Y.%m.%d %H:%M:%S'),'Initializing mulilayer neural network classifier...'))
		#self.x = tf.placeholder(np.int32, [None, 3072])
		#self.y = tf.placeholder(np.int32, [None, 10])
		self.x = tf.placeholder(tf.float32, [None, 3072])
		self.y = tf.placeholder(tf.float32, [None, 10])

		# the size of the different layers were chosen by the slide, that can be found
		# at the https://www.slideshare.net/DaleiLEE/applying-neural-networks-to-cifar10-dataset link
		self.w1 = tf.Variable(tf.random_normal([3072,1024]) )
		self.w2 = tf.Variable(tf.random_normal([1024,600]) )
		self.w3 = tf.Variable(tf.random_normal([600,10]) )

		# biases
		self.b1 = tf.Variable(tf.random_normal([1024]))
		self.b2 = tf.Variable(tf.random_normal([600]))
		self.b3 = tf.Variable(tf.random_normal([10]))
		
		# forward propagation
		self.model = self.forwardPropagation()
		self.predictor = tf.nn.softmax(self.model, dim=1)

	def forwardPropagation(self):
		# this function defines the model of the neural network
		layer1 = tf.nn.relu(tf.matmul(self.x, self.w1)) + self.b1
		layer2 = tf.nn.relu(tf.matmul(layer1, self.w2)) + self.b2
		layerOut = tf.nn.relu(tf.matmul(layer2, self.w3)) + self.b3
		return layerOut

	def train(self, Xin, Yin, epochs=1, batchSize=200):
		X = Xin.reshape(Xin.shape[0], 3072)
		Y = np.zeros([len(Yin), 10])
		for i in range(len(Y)):
			Y[i, Yin[i]] = 1


		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.model) )
		self.updates = tf.train.GradientDescentOptimizer(0.01).minimize(self.cost)
		init = tf.global_variables_initializer()

		with tf.Session() as session:
			session.run(init)

			for epoch in range(epochs):
				for i in range(math.ceil(X.shape[0]/batchSize)):
					batchStart = i * batchSize
					batchEnd = batchStart + batchSize
					if batchEnd > len(X): batchEnd = len(X)
					if i%50 == 0: print('[%s]: epoch %3d, element %3d'%(datetime.now().strftime('%Y.%m.%d %H:%M:%S'), epoch, i))
					print(Y[batchStart:batchEnd])
					_,c = session.run(self.updates, feed_dict={self.x: X[batchStart:batchEnd,:],
					self.y:
					Y[batchStart:batchEnd]})

			#for epoch in range(epochs):
			#	for i in range(X.shape[0]):
			#		if i%500 == 0: print('[%s]: epoch %3d, element %3d'%(datetime.now().strftime('%Y.%m.%d %H:%M:%S'), epoch, i))
			#		print(X[i,:].shape)
			#		print(X[i,:].transpose())
			#		print(X[i,:].transpose().shape)
			#		print(Y[i])
			#		_,c = session.run(self.updates, feed_dict={self.x: np.mat(X[i]), self.y: np.mat(Y[i])})
	
	def predict(self, Xin):
		X = Xin.reshape(Xin.shape[0], 3072)
		init = tf.global_variables_initializer()
		with tf.Session() as session:
			session.run(init)
			prediction = session.run(self.predictor, feed_dict={self.x: X, })
			
		print(np.argmax(prediction, axis=1))
		
if __name__ == '__main__':
	from cifar10reader import loadInputs

	Xtr, Ytr, Xte, Yte, dictionary = loadInputs('cifar-10-batches-py')
	mlp = classifierMLP()
	mlp.train(Xtr[0:10000,:],Ytr[0:10000])
	mlp.predict(Xte[0:100,:])
	#accuracy = np.mean(Yte[0:guess.shape[0]] == guess )
	#print('accuracy: ', accuracy)
