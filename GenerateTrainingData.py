import numpy
import matplotlib.pyplot as plt

# Functions
def graph(w_0, w_1, w_2):
    x = numpy.linspace(-1, 1)
    y = numpy.array([float((-w_0-i*w_1)/(w_2)) for i in x])
    plt.plot(x, y, 'r--', label='Boundary')

# Parameters
w_0 = numpy.random.uniform(-0.25, 0.25)
w_1 = numpy.random.uniform(-1, 1)
w_2 = numpy.random.uniform(-1, 1)
W = numpy.array([w_0, w_1, w_2])
W_Prime = numpy.random.uniform(-1, 1, [1, 3])
X = numpy.random.uniform(-1, 1, [1000, 2])
Class_Positive = numpy.array([[0, 0]])
Class_Negative = numpy.array([[0, 0]])

for i in range(0, 1000):
    a = numpy.array([1, X[i, 0], X[i, 1]])
    if numpy.dot(a, W) >= 0:
        Class_Positive = numpy.concatenate((Class_Positive, [[X[i, 0], X[i, 1]]]), axis = 0)
    else:
        Class_Negative = numpy.concatenate((Class_Negative, [[X[i, 0], X[i, 1]]]), axis=0)

# Generate the Training Data and Store them in txt file
numpy.savetxt('ClassPositive.txt', Class_Positive[1:,:], delimiter=' ')
numpy.savetxt('ClassNegative.txt', Class_Negative[1:,:], delimiter=' ')
numpy.savetxt('X.txt', X, delimiter=' ')
numpy.savetxt('Weights.txt', W, delimiter=' ')
numpy.savetxt('InitialisationWeights.txt', W_Prime, delimiter=' ')

# Plot
fig, ax = plt.subplots()
plt.plot(Class_Positive[1:, 0], Class_Positive[1:, 1], 'r+', label='Postive Class Data')
plt.plot(Class_Negative[1:, 0], Class_Negative[1:, 1], 'bx', label='Negative Class Data')
graph(w_0, w_1, w_2)
ax.legend()
plt.show()