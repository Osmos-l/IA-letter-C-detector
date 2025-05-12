import numpy as np

class Perceptron:
    def __init__(self, epochs, learning_rate, weights=None):
        self.learning_rate = learning_rate
        self.epochs = epochs 
        self.weights = weights if weights is not None else []
        self.loss_before = None
        self.loss_after = None

    def sign(self, s):
        if s > 0:
            return 1
        else:
            return -1

    def dot(self, x):
        s = 0
        for i in range(len(self.weights)):
            s += self.weights[i] * x[i]
        return s

    def predict(self, x):
        s = self.dot(x)
        return self.sign(s)

    def cost(self, x, y):
        yp = self.predict(x)
        c = (y - yp) ** 2
        return c

    def loss(self, data):
        P = len(data)
        s = 0
        for entry in data:
            x = np.array(entry['matrix']).flatten()
            y = entry['is_c']
            s += self.cost(x, y)
        return s / P

    def learn(self, data):
        self.loss_before = self.loss(data)

        for epoch in range(self.epochs):
            for entry in data:
                x = np.array(entry['matrix']).flatten()
                y = entry['is_c']

                yp = self.predict(x)
                # Adjust weights
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * (y - yp) * x[i]
        self.loss_after = self.loss(data)

        print(f"[Simple perceptron] - W: {self.weights}")
        print(f"Loss before: {self.loss_before}")
        print(f"Loss after: {self.loss_after}")
        return self.weights

# Example usage:
# if __name__ == "__main__":
#     import numpy as np
#     from benchmark import benchmark
#     from matrices import data
#
#     # Hyperparameters
#     epochs = 10000            # Number of epochs
#     weights = np.zeros(25)   # Neuron weights (0 by default)
#     learning_rate = 0.01     # Weight update size
#
#     # Create a Perceptron instance
#     perceptron = Perceptron(epochs, learning_rate, weights.copy())
#     perceptron.learn(data)