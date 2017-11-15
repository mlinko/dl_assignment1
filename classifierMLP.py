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
        self.x = tf.placeholder(tf.float32, [None, 3072])
        self.y = tf.placeholder(tf.float32, [None, 10])

        # the size of the different layers were chosen by the slide, that can be found
        # at the https://www.slideshare.net/DaleiLEE/applying-neural-networks-to-cifar10-dataset link
        # after that the performance was too bad, so i have modified it
        self.w1 = tf.Variable(tf.random_normal([3072, 2048]), name='w1', trainable=True, dtype=tf.float32)
        self.w2 = tf.Variable(tf.random_normal([2048, 1024]), name='w2', trainable=True, dtype=tf.float32)
        self.w3 = tf.Variable(tf.random_normal([1024, 512]), name='w3', trainable=True, dtype=tf.float32)
        self.w4 = tf.Variable(tf.random_normal([512, 10]), name='w4', trainable=True, dtype=tf.float32)

        # biases
        self.b1 = tf.Variable(tf.random_normal([2048]), name='b1', trainable=True, dtype=tf.float32)
        self.b2 = tf.Variable(tf.random_normal([1024]), name='b2', trainable=True, dtype=tf.float32)
        self.b3 = tf.Variable(tf.random_normal([512]), name='b3', trainable=True, dtype=tf.float32)
        self.b4 = tf.Variable(tf.random_normal([10]), name='b4', trainable=True, dtype=tf.float32)
        
        # forward propagation
        self.model = self.forwardPropagation()
        self.predictor = tf.nn.softmax(self.model, dim=1)

        self.GAMMA = 0.0005
        self.learningRate = 0.0005


    def shuffleData(self,x,y):
        assert len(x) == len(y)
        indexes = np.arange(len(x))
        np.random.shuffle(indexes)
        xout = x.copy()
        yout = y.copy()
        for i in indexes:
            xout[i] = x[ indexes[i]]
            yout[i] = y[ indexes[i]]
        return xout, yout
            

    def forwardPropagation(self):
        # this function defines the model of the neural network
        layer1 = tf.nn.relu(tf.matmul(self.x, self.w1) + self.b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, self.w2) + self.b2)
        layer3 = tf.nn.relu(tf.matmul(layer2, self.w3) + self.b3)
        # I leave this wrong line here commented, to remind me
        # NEVER EVER USE ANY FUNCTION ON THE OUTPUT LAYER!!!!
        # layerOut = tf.nn.relu(tf.matmul(layer3, self.w4) + self.b4)
        layerOut = tf.matmul(layer3, self.w4) + self.b4 
        return layerOut

    def train(self, Xin, Yin, Xtein, Yte, epochs=1, batchSize=100):
        X = Xin.reshape(Xin.shape[0], 3072)
        Xte = Xtein.reshape(Xtein.shape[0], 3072)
        Y = np.zeros([len(Yin), 10], dtype=np.float32)
        for i in range(len(Y)):
            Y[i, Yin[i]] = 1

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.model) )
        self.l2_cost = self.GAMMA * sum(tf.nn.l2_loss(layer) for layer in tf.trainable_variables() )
        self.updates = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.cost + self.l2_cost)
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
                        loss = session.run(self.cost, feed_dict={self.x: X[batchStart:batchEnd, :],
                                                                 self.y: Y[batchStart:batchEnd]})
                        print('[%s]: epoch %3d, batch %3d, loss %f' % (datetime.now().strftime('%Y.%m.%d %H:%M:%S'), epoch, i, loss))

                    c = session.run(self.updates, feed_dict={self.x: X[batchStart:batchEnd,:], self.y: Y[batchStart:batchEnd]})
                #for layer in tf.trainable_variables():
                #    if 'w' in layer.name:
                #        print(layer.name)
                #        print(session.run(layer))
                guess = np.argmax( session.run(self.predictor, feed_dict={self.x: Xte }), axis=1)
                accuracy = np.mean(Yte == guess)
                print('[%s]: epoch %3d, accuracy %2.1f %%'%(datetime.now().strftime('%Y.%m.%d %H:%M:%S'), epoch, accuracy *100))
                savePath = saver.save(session,'mlp/3_epoch%d/model.ckpt'%epoch)
                X, Y = self.shuffleData(X,Y)
    
    def predict(self, Xin, weightsFile):

        X = Xin.reshape(Xin.shape[0], 3072)
        #init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as session:
            #session.run(init)
            saver.restore(session, weightsFile)
            prediction = session.run(self.predictor, feed_dict={self.x: X})
            #for layer in tf.trainable_variables():
            #    if 'w' in layer.name:
            #        print(layer.name)
            #        print(session.run(layer))
        return np.argmax(prediction, axis=1)
        
if __name__ == '__main__':
    from cifar10reader import loadInputs

    Xtr, Ytr, Xte, Yte, dictionary = loadInputs('cifar-10-batches-py')
    mlp = classifierMLP()
    #mlp.train(Xtr,Ytr, Xte, Yte, epochs=500)
    guess = mlp.predict(Xte, 'mlp/2_epoch465/model.ckpt')
    print(guess)
    accuracy = np.mean(Yte == guess )
    print('accuracy: ', accuracy)
