# Importing MNIST data set using sklearn
# pyplot and patches are used for plotting graphs

from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# The data that is imported is copied into the below variable

custom_data_home = "./"
mnist = fetch_mldata('MNIST original', data_home=custom_data_home)

# Variables used to define the number of traning and testing examples from the sample
traning_examples = 600
testing_examples = 100
total_examples = 60000
class_labels = 10

# Uses array from numpy for the mnist dat to convert to 32bit integer
mnist_data = np.array(mnist.data.shape, dtype=np.int32)
mnist_data = np.copy(mnist.data.astype(np.int32))


def knn_classifier():
    k = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    # the imported data has labes that are orders
    # shuffle is used to shuffle the input image data labels
    shuffle = np.arange(total_examples)
    np.random.shuffle(shuffle)
    # Used to intilize zeros in the traning data variable and take first 6000 samples
    traning_data = np.zeros((traning_examples, mnist.data.shape[1]), dtype=np.int32)
    traning_data[:] = mnist_data[shuffle[:traning_examples]]
    traning_target = mnist.target[shuffle[:traning_examples]]
    # Used to intilize zeros in the testing data variables and the last 1000 samples
    testing_data = np.zeros((testing_examples, mnist.data.shape[1]), dtype=np.int32)
    testing_data[:] = mnist_data[shuffle[(total_examples - testing_examples): total_examples]]
    testing_target = mnist.target[shuffle[(total_examples - testing_examples): total_examples]]
    # Used to initilize zeros in the distance variable and convert it into 32 bit integer data type
    euclidean_distance = np.zeros((traning_examples, traning_examples), dtype=np.int32)
    euclidean_distance_copy = np.zeros((testing_examples, traning_examples), dtype=np.int32)
    # temporary variables to store ecudilian distance
    temporary = np.empty_like(euclidean_distance)
    temporary_copy = np.empty_like(euclidean_distance_copy)
    # Clasification varaible with traning and testing examples and their corrosponding labels
    classification = np.zeros((traning_examples, class_labels))
    classification_copy = np.zeros((testing_examples, class_labels))
    # Calcualtes ecudilian distance for traning data
    for i in range(0, traning_examples):
        euclidean_distance[i:i + 1, :] = \
            np.sqrt(np.sum(np.square(traning_data[:traning_examples, :] - traning_data[i, :]), axis=1))
    # Calculates eculidian data for testing data
    for i in range(0, testing_examples):
        euclidean_distance_copy[i:i + 1, :] = \
            np.sqrt(np.sum(np.square(traning_data[:, :] - testing_data[i, :]), axis=1))
    # Sorting the calcualted distance by size
    sorted_indices = np.argsort(euclidean_distance)
    sorted_indices_copy = np.argsort(euclidean_distance_copy)

    labels = traning_target[:][sorted_indices]
    labels_copy = traning_target[:][sorted_indices_copy]

    for i in range(0, traning_examples):
        temporary[i, :] = euclidean_distance[i, :][sorted_indices[i]]

    for i in range(0, testing_examples):
        temporary_copy[i, :] = euclidean_distance_copy[i, :][sorted_indices_copy[i]]

    euclidean_distance[:, :] = temporary[:, :]
    euclidean_distance_copy[:, :] = temporary_copy[:, :]
    # Used to delete the temporary variables created
    np.delete(temporary, np.s_[:], 1)
    np.delete(temporary_copy, np.s_[:], 1)

    traning_error = np.zeros((1, len(k)), dtype=np.float)
    testing_error = np.zeros((1, len(k)), dtype=np.float)

    for loop in k:
        for i in range(0, traning_examples):
            for j in range(0, loop):
                index = int(labels[i, j])
                classification[i, index] += 1

        for i in range(0, testing_examples):
            for j in range(0, loop):
                index = int(labels_copy[i, j])
                classification_copy[i, index] += 1

        temporary = np.argsort(classification)
        temporary_copy = np.argsort(classification_copy)

        for i in range(0, traning_examples):
            if temporary[i, class_labels - 1] != traning_target[i]:
                traning_error[0, k.index(loop)] += 1

        for i in range(0, testing_examples):
            if temporary_copy[i, class_labels - 1] != testing_target[i]:
                testing_error[0, k.index(loop)] += 1

        traning_error[0, k.index(loop)] /= traning_examples
        testing_error[0, k.index(loop)] /= testing_examples

    # plotting the graphs on same plane for errors vs value of K
    plt.xlabel('Value of K')
    plt.ylabel('Error')
    # red represents the test error
    # green represents the traning error
    red_box = mpatches.Patch(color='red', label='Test Error')
    green_box = mpatches.Patch(color='green', label='Traning Error')

    plt.legend(handles=[red_box,green_box])

    plt.plot(k, testing_error[0], color='red', marker='*')
    plt.plot(k, traning_error[0], color='g')
    plt.title('knn Classifier')
    plt.show()


# Calling knn Classifier() module
knn_classifier()
