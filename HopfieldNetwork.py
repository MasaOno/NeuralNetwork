import numpy as np
import copy

class HopfieldNetwork:

    def __init__(self, patterns):
        # Initialize
        self.patterns = np.array(patterns)
        self.patternShape = self.patterns.shape
        self.weights = np.zeros((self.patternShape[1], self.patternShape[1]))

    def setWeights(self):
        # Train the weights
        for pattern in self.patterns:
            for i in xrange(len(self.weights)):
                for j in xrange(len(self.weights)):
                    self.weights[i][j] += (pattern[i]) * (pattern[j])
        self.weights = self.weights / self.patternShape[0]
        return 1

    def trainPattern(self, count):
        # Test random patterns until it converges
        # Initialize random test patterns
        testPattern = []
        for i in xrange(count):
            tempPattern = [2 * np.random.randint(2) - 1 for x in range(len(self.patterns[0]))]
            testPattern.append(tempPattern)
        testPattern = np.array(testPattern)
        # Patterns of each step until convergence
        returnPattern = []
        returnPattern.append(copy.copy(testPattern))
        f = np.vectorize(lambda x: -1 if x<0 else +1)
        # First evolution
        for z in xrange(len(testPattern)):
            testPattern[z] = f(np.dot(self.weights, np.transpose(testPattern[z])))
        returnPattern.append(copy.copy(testPattern))
        # Iterate evolutions
        count = 0
        print testPattern.tolist()
        print returnPattern[count].tolist()
        while testPattern.tolist() != returnPattern[count].tolist():
            # Update pattern
            for z in xrange(len(testPattern)):
                testPattern[z] = f(np.dot(self.weights, np.transpose(testPattern[z])))
            returnPattern.append(copy.copy(testPattern))
            count += 1
            print testPattern.tolist()
        return returnPattern

    def correctMatrix(self, pattern):
        # Print organized view of the progression of pattern
        for i in xrange(len(pattern[0])):
            for matrix in pattern:
                print matrix[i]
            print "\n"
        return 1

