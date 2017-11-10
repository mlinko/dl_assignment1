#!/usr/bin/python3
import numpy as np

class classifierNN:
	def __init__(self, dictionary):
		pass
	
	def train(self, X, Y):
		self.X = X.reshape(X.shape[0], 3072)
		self.Y = Y

	def predict(self, xi):
		x = xi.reshape(xi.shape[0], 3072)
		y = np.zeros(self.X.shape[0], dtype=int)
		print(x, y)
		for i in range( self.X.shape[0]):
			print( np.abs(self.X - x[i,:]))
#			distances = np.sum( np.abs(self.X - x[i,:]), axis=1 )
#			y[i] = self.Y[np.argmin(distances)]
			


if __name__ == '__main__':
	from cifar10reader import loadInputs

	Xtr, Ytr, Xte, Yte, dictionary = loadInputs('cifar-10-batches-py')
	nn = classifierNN(dictionary)
	nn.train(Xtr,Ytr)
	nn.predict(Xte)

			
			
		

