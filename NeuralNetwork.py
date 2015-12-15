import numpy as np
# import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivSigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def initializeCostFunction(var):
    function = []
    for layer in var:
        function.append(np.zeros(layer.shape))
    return function

def costDerivative(activation, y):
    return 2 * (activation - y)

def plotError(error):
    errorplot = plt.figure()
    ax = errorplot.add_subplot(111)
    errorplot.subplots_adjust(top=0.85)
    ax.set_title('Error')
    plt.plot(error)
    plt.show()

class NeuralNetwork(object):

    def __init__(self, structure):
        # Initialize the bias and weights with the network structure
        self.struct = structure
        self.layers = len(structure)
        self.bias = []
        for x in structure[1:]:
            self.bias.append(np.random.randn(x, 1))
        self.weights = []
        for x, y in zip(structure[:-1], structure[1:]):
            self.weights.append(np.random.randn(y,x))

    def singleForwardPass(self, data):
        # Pass the data and return the output
        for b, w in zip(self.bias, self.weights):
            data = sigmoid(np.dot(w, data) + b)
        return data

    def backPropagation(self, x, y, lmbda):
        # Calculate the difference and return the cost
        costB = initializeCostFunction(self.bias)
        costW = initializeCostFunction(self.weights)
        # Move forward
        activationLayer = x
        activations = [x]
        node = []
        for b, w in zip(self.bias, self.weights):
            node.append(np.dot(w, activationLayer) + b)
            activationLayer = sigmoid(np.dot(w, activationLayer) + b)
            activations.append(activationLayer)
        # Calculate for the initial deriv
        delta = (costDerivative(activations[-1], y) + lmbda) * derivSigmoid(node[-1])
        costB[-1] = delta
        costW[-1] = np.dot(delta, activations[-2].transpose())
        # Move backwards through each layer and do the same
        for layer in xrange(-2, -1 * self.layers, -1):
            n = node[layer]
            delta = np.dot(self.weights[layer + 1].transpose(), delta) * derivSigmoid(n)
            costB[layer] = delta
            costW[layer] = np.dot(delta, activations[layer - 1].transpose())
        return (costB, costW)

    def trainNetwork(self, trainData, learningRate, iterations, lmbda, testData):
        # Train the training data and give error outputs
        error = []
        for iteration in xrange(iterations):
            costB = initializeCostFunction(self.bias)
            costW = initializeCostFunction(self.weights)
            for x, y in trainData:
                # run backProp to find the delta
                deltaB, deltaW = self.backPropagation(x, y, lmbda)
                # calculate the error
                costB = [singleB + singleDeltaB for singleB, singleDeltaB in zip(costB, deltaB)]
                costW = [singleW + singleDeltaW for singleW, singleDeltaW in zip(costW, deltaW)]
            # update the bias and weights
            self.bias = [b - (learningRate / len(trainData)) * c for b, c in zip(self.bias, costB)]
            self.weights = [w - (learningRate / len(trainData)) * c for w, c in zip(self.weights, costW)]
            # Print current status
            print 'Iteration ' + str(iteration) + ": " + str(self.accuracy(testData))
            error.append(self.accuracy(testData) / float(len(testData)))
        plotError(error)

    def accuracy(self, data):
        results = []
        for x, y in data:
            r = (np.argmax(self.singleForwardPass(x)), y)
            results.append(r)
        return sum(int(x == y) for (x, y) in results)

