import numpy
import matplotlib.pyplot as plt

# Load The Training Data
W = numpy.loadtxt('Weights.txt') # Weights
W_Prime = numpy.loadtxt('InitialisationWeights.txt') # Randomly Generated Initial Weight
ClassPositive = numpy.loadtxt('ClassPositive.txt') # Set S1
ClassNegative = numpy.loadtxt('ClassNegative.txt') # Set S0
X = numpy.loadtxt('X.txt') # Set S

# Parameters
eta = 0.1 # Learning Rate
e = 0
Epoch = numpy.array([0])
Error = numpy.array([]) # No.of Wrong Classifications

# Functions
def eval_perceptron(ClassPositive, ClassNegative, W):
    m = 0
    row, col = ClassPositive.shape
    for i in range(0, row):
        a = numpy.array([1, ClassPositive[i, 0], ClassPositive[i, 1]])
        if numpy.dot(a, W) < 0:
            m += 1
    row, col = ClassNegative.shape
    for i in range(0, row):
        a = numpy.array([1, ClassNegative[i, 0], ClassNegative[i, 1]])
        if numpy.dot(a, W) >= 0:
            m += 1
    return m

def update_weights(ClassPositive, ClassNegative, W, eta):
    row, col = ClassPositive.shape
    for i in range(0, row):
        a = numpy.array([1, ClassPositive[i, 0], ClassPositive[i, 1]])
        if numpy.dot(a, W) < 0:
            W = W + eta*a
    row, col = ClassNegative.shape
    for i in range(0, row):
        a = numpy.array([1, ClassNegative[i, 0], ClassNegative[i, 1]])
        if numpy.dot(a, W) >= 0:
            W = W - eta*a
    return W

def graph(w_0, w_1, w_2, a, b):
    x = numpy.linspace(-1, 1)
    y = numpy.array([float((-w_0-i*w_1)/(w_2)) for i in x])
    plt.plot(x, y, a, label=b)

# Main
w_current = W_Prime
m = eval_perceptron(ClassPositive, ClassNegative, w_current)
Error = numpy.concatenate((Error, [m]), axis = 0)
W_Temp = numpy.array([w_current[0], w_current[1], w_current[2]])
print('Number of errors in the Epoch ', e, ': \t', m)
print('Weight Matrix in the Epoch ', e, ': \t', w_current, '\n')
while m > 0:
    e += 1
    w_current = update_weights(ClassPositive, ClassNegative, w_current, eta)
    m = eval_perceptron(ClassPositive, ClassNegative, w_current)
    # Book Keeping
    Error = numpy.concatenate((Error, [m]), axis=0)
    Epoch = numpy.concatenate((Epoch, [e]), axis=0)
    W_Temp = numpy.concatenate((W_Temp, [w_current[0], w_current[1], w_current[2]]), axis=0)
    # Print
    print('Number of errors in the Epoch ', e, ': \t', m)
    print('Weight Matrix in the Epoch ', e, ': \t', w_current, '\n')
print('Actual Weight Matrix: ', W)
print('Trained Final Weight Matrix: ', w_current)
numpy.savetxt('FinalOptimalWeights.txt', w_current)

# Plot
# Plot 1
fig, ax1 = plt.subplots()
ax1.plot(ClassPositive[1:, 0], ClassPositive[1:, 1], 'r+', label='Positive Class Data')
ax1.plot(ClassNegative[1:, 0], ClassNegative[1:, 1], 'bx', label='Negative Class Data')
graph(W[0], W[1], W[2], 'r--', 'Boundary')
#ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Training Data for Linear Classifier with Samples N = 1000')
plt.tight_layout()
# Plot 2
fig, ax2 = plt.subplots()
ax2.plot(ClassPositive[1:, 0], ClassPositive[1:, 1], 'r+', label='Positive Class Data')
ax2.plot(ClassNegative[1:, 0], ClassNegative[1:, 1], 'bx', label='Negative Class Data')
graph(w_current[0], w_current[1], w_current[2], 'g--', 'Boundary')
#ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('New Linear Classifier after Training has been done on Samples N = 1000')
fig.text(0.39, 0.03, r'Learning Rate $\eta = 0.1$', ha='center')
plt.tight_layout()
# Plot 3
fig, ax3 = plt.subplots()
ax3.plot(Epoch, Error, label='Misclassifications')
ax3.legend()
plt.title(r'Epoch VS No.of Misclassifications for Learning Rate $\eta = 0.1$')
plt.xlabel('Epoch')
plt.ylabel('No.of Misclassifications')
plt.tight_layout()
plt.show()





