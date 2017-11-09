#!/usr/bin/python3
import pickle
import numpy as np
import os

def unpickle(batchFile):
# this function is provided on the official site
# of the cifar-10 database /www.cs.toronto.edu/~kriz/cifar.html/
	with open(batchFile, 'rb') as f:
		data = pickle.load(f, encoding='latin1')
	return data

def loadBatch(batchFile, numberOfCase):
	data = unpickle(batchFile)
	inputs = data['data'].reshape(( numberOfCase, 3, 1024))
	labels = np.array(data['labels'])
	return inputs, labels
	
def loadInputs(inputDir):
	metaFile = 'batches.meta'
	trainFiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
	testFile = 'test_batch'

	metadata = unpickle(os.path.join(inputDir, metaFile))

	Xtr = np.zeros(shape=[metadata['num_cases_per_batch'] * len(trainFiles) , 3, metadata['num_vis']//3 ], dtype=int)
	Ytr = np.zeros([metadata['num_cases_per_batch'] * len(trainFiles)], dtype=int)
	
	for i in range(len(trainFiles)):
		batchStart = i * metadata['num_cases_per_batch']
		batchEnd = (i+1) * metadata['num_cases_per_batch']
		x, y = loadBatch(os.path.join(inputDir,trainFiles[i]), metadata['num_cases_per_batch'])
		Xtr[batchStart:batchEnd ,:] = x
		Ytr[batchStart:batchEnd] = y
	
	Xte, Yte = loadBatch(os.path.join(inputDir, testFile), metadata['num_cases_per_batch'])

	return Xtr, Ytr, Xte, Yte, metadata['label_names']

if __name__ == '__main__':
	print(loadInputs('cifar-10-batches-py'))

