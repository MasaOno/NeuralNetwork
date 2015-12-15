import numpy as np
import heapq
import matplotlib.pyplot as plt

# Plot the false-pos errors
def plot(error, y):
    errorplot = plt.figure()
    ax = errorplot.add_subplot(111)
    errorplot.subplots_adjust(top=0.85)
    ax.set_title('Error')
    plt.plot(error, y)
    plt.show()

class ReinforcementLearning(object):

    def __init__(self, numInputs, numStimuli, numReward, sparcity):
    	# Initialize
        self.numInputs = numInputs
        self.numStimuli = numStimuli
        self.numReward = numReward
        self.sparcity = sparcity
        self.matrix = np.ceil(np.random.random((numInputs, numStimuli)) - sparcity)
        self.weights = np.zeros([1, self.numInputs])
        self.rewarded = np.zeros(self.numStimuli)
        for i in np.random.choice(self.numStimuli, numReward, replace = False):
            self.rewarded[i] = 1

    def simulate(self, iterations):
        # For every iteration...
        for iteration in xrange(iterations):
            # Find a new set of neurons that fire
            firedNeurons = np.zeros(self.numInputs)
            for i in np.random.choice(self.numInputs, self.numInputs - self.sparcity, replace = False):
                firedNeurons[i] = 1
            # Update weigh if both rewarded and fired
            for k in xrange(self.numInputs):
                for j in xrange(self.numStimuli):
                    if firedNeurons[k] == self.rewarded[j]:
                        # Update the weight!
                        self.weights[0][k] = self.weights[0][k] + 2 * (1 - self.sparcity) - self.matrix[k][j] * (1 - self.sparcity)
        # Find the output for each input
        output = []
        for k in xrange(self.numStimuli):
            temp = 0
            for j in xrange(self.numInputs):
                temp += self.weights[0][j] * self.matrix[j][k]  
            output.append(temp)
        correctedOutput = np.zeros(self.numInputs)
        posIndex = heapq.nlargest(self.numReward, output)
        for item in posIndex:
            correctedOutput[posIndex.index(item)] = 1
        return (self.rewarded, correctedOutput)

