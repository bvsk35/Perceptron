# This files Tests the Neural Network

# Import required Libraries
import numpy
from LoadData import load_test_images, load_test_labels

# Initialisations
W = numpy.loadtxt('FinalOptimalWeights.txt')
Images = load_test_images('t10k-images-idx3-ubyte.gz')
Labels = load_test_labels('t10k-labels-idx1-ubyte.gz')

# Functions
def eval_perceptron(W, Images, Labels):
    m = 0
    row, col = Images.shape
    for i in range(0, row):
        a = numpy.dot(W, Images[i,:])
        digit = numpy.argmax(a)
        if digit != Labels[i]:
            m += 1
    return m

# Main Loop
row, col = Images.shape
errors = eval_perceptron(W, Images, Labels)
error_percentage = float(errors/row)*100
s = numpy.array([errors, row, error_percentage])
numpy.savetxt('ErrorPercentage.txt', s)