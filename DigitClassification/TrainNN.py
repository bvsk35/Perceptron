# This file Trains the Neural Network

# Import required libraries
import numpy
import matplotlib.pyplot as plt
from LoadData import load_training_labels, load_training_images
from LoadData import InitialWeights

# Initialisation
eta = 0.7 # Learning Rate
N = 1000 # No.of Training Samples
epsilon = 0 # Error Ratio
e = 0 # Epoch
Epoch = numpy.array([0]) # Storing Epoch for plotting
Error = numpy.array([]) # No.of Wrong Classifications
Images = load_training_images('train-images-idx3-ubyte.gz')
Labels = load_training_labels('train-labels-idx1-ubyte.gz')
if numpy.DataSource().exists('InitialWeights.txt'):
    W_Prime = numpy.loadtxt('InitialWeights.txt') # Load the Initial Weights
else:
    W_Prime = InitialWeights()
    numpy.savetxt('InitialWeights.txt', W_Prime) # Generate the Weights and save them

# Functions
# Counts Misclassifications
def eval_perceptron(W, N, Images, Labels):
    m = 0
    for i in range(0, N):
        a = numpy.dot(W, Images[i,:])
        digit = numpy.argmax(a)
        if digit != Labels[i]:
            m += 1
    return m
# Update Weights
def update_weights(W, N, eta, Images, Labels):
    for i in range(0, N):
        d = numpy.zeros(10)
        d[Labels[i]] = 1
        a = numpy.heaviside(numpy.dot(W, Images[i,:]), 1)
        W = W + numpy.outer(eta*(d-a), Images[i,:])
    return W

# Main Loop
m = eval_perceptron(W_Prime, N, Images, Labels)
W_current = update_weights(W_Prime, N, eta, Images, Labels)
Error = numpy.concatenate((Error, [m]), axis = 0)
while float(m/N) > epsilon:
    e += 1
    m = eval_perceptron(W_current, N, Images, Labels)
    W_current = update_weights(W_current, N, eta, Images, Labels)
    # Book Keeping
    Error = numpy.concatenate((Error, [m]), axis=0)
    Epoch = numpy.concatenate((Epoch, [e]), axis=0)
    # Print
    print('Number of errors in the Epoch ', e, ': \t', m)
numpy.savetxt('FinalOptimalWeights.txt', W_current)

# Plot
fig, ax = plt.subplots()
ax.plot(Epoch, Error, label='Misclassifications')
ax.legend()
plt.title(r'Epoch VS No.of Misclassifications for Learning Rate $\eta = 0.7$ and for Samples N = 60000'
          r' and Threshold $\epsilon = 0.13$')
plt.xlabel('Epoch')
plt.ylabel('No.of Misclassifications')
#plt.tight_layout()
plt.show()