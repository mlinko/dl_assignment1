# dl_assignment1
## Tasks
1. Get familiar with the CIFAR-10 database, and look after methods how to read and process it efficiently.
2. Implement 3 classifier algorithms that will return the prediction of the computer (implement them as functions in individual modules using TensorFlow) - specifications: input: image [H, W, 3] & output: index
   - Write a classifier that uses Nearest Neighbor search on the training set, and returns the closest match's label as an index.
   - Train a Multi Layer Perceptron on the training set previously, save the weights and biases, and when classifier function is called just initialize the network with the pretrained weights, forward the input, check the highest activation on the last layer and return the index of it.
   - Train a Convolutional Neural Network on the training set previously, save the weights and biases, and when classifier function is called just initialize the network with the pretrained weights, forward the input, check the highest activation on the last layer and return the index of it.
3. Write a python script that takes in an image and feeds it to classifierNN, classifierMLP, classifierCNN, and evaluate the correctness of these function on the test / training dataset
4. Optional: Create your own database using your device of 3 (at least 20 samples per class) different classes, and perform Task 1, 2, 3 on it as well.
