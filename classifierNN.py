#!/usr/bin/python3
import numpy as np
from datetime import datetime

class classifierNN:
	def __init__(self, dictionary):
		pass
	
	def train(self, X, Y):
		self.X = X.reshape(X.shape[0], 3072)
		self.Y = Y

	def predict(self, xi):
		x = xi.reshape(xi.shape[0], 3072)
		y = np.zeros(x.shape[0], dtype=int)
		for i in range( x.shape[0]):
			if i%2 == 0: print('[%s]: %5d th element'%(datetime.now().strftime('%Y.%m.%d %H:%M:%S'),i))
			distances = np.sum( np.abs(self.X - x[i,:]), axis=1 )
			y[i] = self.Y[np.argmin(distances)]
		return y
			


if __name__ == '__main__':
	from cifar10reader import loadInputs

	Xtr, Ytr, Xte, Yte, dictionary = loadInputs('cifar-10-batches-py')
	nn = classifierNN(dictionary)
	nn.train(Xtr,Ytr)
	guess = nn.predict(Xte)
	guess.tofile('1nn_guess')
	accuracy = np.mean(Yte[0:guess.shape[0]] == guess )
	print('accuracy: ', accuracy)
