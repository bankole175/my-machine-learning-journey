import numpy as np

np.random.seed(42)

# Dataset — XOR problem (cannot be solved by linear models)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Network architecture
# Input layer:  2 neurons  (2 features)
# Hidden layer: 4 neurons
# Output layer: 1 neuron   (binary classification)

# Initialise weights — small random values
W1 = np.random.randn(2, 4) * 0.1   # Shape (2, 4)
b1 = np.zeros((1, 4))               # Shape (1, 4)

W2 = np.random.randn(4, 1) * 0.1   # Shape (4, 1)
b2 = np.zeros((1, 1))               # Shape (1, 1)