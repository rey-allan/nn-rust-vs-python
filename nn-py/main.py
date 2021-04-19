"""
Structure of a Neural Network to learn the `XOR` function
- 2 input features
- 2 hidden neurons
- 1 output neuron
"""
import numpy as np
import random
from time import time


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


if __name__ == '__main__':
    # Generate dataset of `XOR` examples
    # We define the array as suggested by Andrew Ng: features are rows, examples are columns
    # This makes the computation of the linear combination match the math closer: W^T * X
    x = np.array([[0., 0., 1., 1.], [0., 1., 0., 1.]])
    y = np.array([[0., 1., 1., 0.]])
    n = x.shape[1]

    seeds = [2, 10, 24, 45, 98, 120, 350, 600, 899, 1000]
    runtimes = []

    # Execute the training with each different seed and time them
    for seed in seeds:
        print(f"Running with seed: {seed}")
        random.seed(seed)
        start = time()

        # Initialize weights and biases
        # Remember: they have to be initialized to random values in order to break symmetry!
        # Input to hidden
        w0 = np.random.rand(2, 2)
        b0 = np.random.rand(2, 1)
        # Hidden to output
        w1 = np.random.rand(2, 1)
        b1 = np.random.rand(1, 1)

        # Train using gradient descent
        epochs = 100000
        alpha = 0.5

        for _ in range(epochs):
            # Forward propagation
            a1 = sigmoid(np.dot(w0.T, x) + b0)
            y_hat = sigmoid(np.dot(w1.T, a1) + b1)

            # Loss (MSE) multiplied by a 1/2 term to make the derivative easier
            _loss = (1. / (2. * n)) * np.sum((y_hat - y) ** 2)

            # Backpropagation
            dy_hat = (y_hat - y) / float(n)
            dz2 = (y_hat * (1. - y_hat)) * dy_hat
            dw1 = np.dot(a1, dz2.T)
            db1 = np.sum(dz2, axis=1, keepdims=True)
            dz1 = (a1 * (1. - a1)) * np.dot(w1, (dy_hat * (y_hat * (1. - y_hat))))
            dw0 = np.dot(x, dz1.T)
            db0 = np.sum(dz1, axis=1, keepdims=True)

            # Weight and bias update
            # We average the gradients so that we can use a larger learning rate
            w0 = w0 - alpha * (dw0 / float(n))
            w1 = w1 - alpha * (dw1 / float(n))
            b0 = b0 - alpha * (db0 / float(n))
            b1 = b1 - alpha * (db1 / float(n))

        # Record the time
        runtimes.append(round((time() - start) * 1000.))

    # Save runtimes to a file for further processing
    print('Saving runtimes to `python.txt`')
    with open('data/python.txt', 'w') as f:
        for t in runtimes:
            f.write(str(t) + '\n')
