# This file Loads the Training Images and Labels
# Loads Test Images and Labels
# Generate/Initialize Weight Matrix from randomly chosen points in range [-1, 1]
# Distribution

# Import required functions
import gzip
import numpy

# Functions
def load_training_images(filename):
    with gzip.open(filename) as f:
        a = numpy.frombuffer(f.read(), dtype=numpy.uint8, offset=16).reshape(-1, 784)
        return a

def load_training_labels(filename):
    with gzip.open(filename) as f:
        a = numpy.frombuffer(f.read(), dtype=numpy.uint8, offset=8)
        return a

def load_test_images(filename):
    with gzip.open(filename) as f:
        a = numpy.frombuffer(f.read(), dtype=numpy.uint8, offset=16).reshape(-1, 784)
        return a

def load_test_labels(filename):
    with gzip.open(filename) as f:
        a = numpy.frombuffer(f.read(), dtype=numpy.uint8, offset=8)
        return a
def InitialWeights():
    return numpy.random.uniform(-1, 1, size=(10, 784))