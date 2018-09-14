# KNN-Classifier
K Nearest Neighbors classifier from scratch for image classification using MNIST Data Set. [KNN_Classifier](https://github.com/tarunkolla/KNN-Classifier/blob/master/knnClassifier.py)


## Implementation
No existing class or functions (e.g., sklearn.neighbors.KNeighborsClassifier) have been used.

#### Data Set:
MNIST data set consisting of 60000 examples where each example is a hand written digit. 
Each example includes 28x28 grey-scale pixel values as features and a categorical class label out of 0-9. 

Data set can be manually download the dataset from Dr. Yann Lecun’s [webpage](http://yann.lecun.com/exdb/mnist/) or automatically import it from libraries/packages (e.g., as done in [section 5.9](http://scikitlearn.org/stable/datasets/index.html) for sklearn in Python).

A detailed description of the data has also been listed in the above link.

#### Classifier:
Euclidean distance (L2 Norm) has been used to determine the distance.
k = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99].
Data Set has been imported using sklearn.

On the original data set, the first 6,000 examples for training, and the last 1,000 examplesfor testing have been used. 
This can be altered by changing values of *traning_examples , testing_examples*

#### Plot curves for training and test errors: 
The curve shows training/test error (which is equal to 1.0-accuracy) vs. the value of K. 
11 points for the curve, using K = 1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99 have been ploted.
The error curves for training error and test error have been ploted in the same figure.
![Result](https://github.com/tarunkolla/KNN-Classifier/blob/master/result.png)


# Classification with Tensorflow:

Used sequential model with 2 layered neural network with each having 128 neurons on the same dataset. [ImageClassification](https://github.com/tarunkolla/KNN-Classifier/blob/master/imageClassification.py)



