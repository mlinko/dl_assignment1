#!/usr/bin/python3
import numpy as np
from datetime import datetime

class classifierNN_nontf:
	# this class is a Nearest Neighbour classifier, that uses only numpy
	# made only for fun
	def __init__(self, dictionary):
		pass
	
	def train(self, X, Y):
		# here by 'training' I mean adding the training set to the class
		self.X = X.reshape(X.shape[0], 3072)
		self.Y = Y

	def predict(self, xi):
		# prediciton is simply findng the smallest distance between the 
		# train set and the actual test element, and add the label of the
		# nearest object to the actual test element
		x = xi.reshape(xi.shape[0], 3072)
		y = np.zeros(x.shape[0], dtype=int)
		for i in range( x.shape[0]):
			if i%250 == 0: print('[%s]: %5d th element'%(datetime.now().strftime('%Y.%m.%d %H:%M:%S'),i))
			distances = np.sum( np.abs(self.X - x[i,:]), axis=1 )
			y[i] = self.Y[np.argmin(distances)]
		return y
			

import tensorflow as tf

class classifierNN:
	def __init__(self):
		# graph inputs for the tensorflow
		print('[%s]: %s'%(datetime.now().strftime('%Y.%m.%d %H:%M:%S'),'Initializing nearest neighbour classifier...'))
		self.xtr = tf.placeholder(np.int32, [None, 3072])
		self.xte = tf.placeholder(np.int32, [3072])

		# On the http://cs231n.github.io/classification/ website
		# the L1 distance is being used, which I am implementing here too
		self.distance = tf.reduce_sum(tf.abs(tf.add(self.xtr, tf.negative(self.xte))))
		# prediciton is based on the nearest object
		self.predictor = tf.argmin(self.distance, axis=0)
		# the next line initializes hte variables (assigns their )
		self.init = tf.global_variables_initializer()
		
	def train(self, Xtr, Ytr):
		# real training cannot really be implemented here, so
		# I just pass the train set to the object
		self.Xtr = Xtr.reshape(Xtr.shape[0], 3072)
		self.Ytr = Ytr

	def predict(self, X ):
		# prediction is the variable we will return with
		Xte = X.reshape(X.shape[0], 3072)
		prediction = np.zeros([len(Xte)])
		with tf.Session() as session:
			# iterating trough the elements of the trains set, and
			# with the help of the predict operator we are finding the index of the nearest
			# neighbour
			# the prediction will be the class number of the nearest neighbour for the actal element
			for i in range(len(Xte)):
				neareastNeighbour = session.run(self.predictor, feed_dict={self.xtr: self.Xtr, self.xte:Xte[i,:]})
				prediction[i] = self.Ytr[nearestNeighbour]
		return prediction
				
if __name__ == '__main__':
	from cifar10reader import loadInputs

	Xtr, Ytr, Xte, Yte, dictionary = loadInputs('cifar-10-batches-py')
	nn = classifierNN()
	nn.train(Xtr[0:10000,:],Ytr[0:10000])
	guess = nn.predict(Xte[0:100,:])
	accuracy = np.mean(Yte[0:guess.shape[0]] == guess )
	print('accuracy: ', accuracy)
