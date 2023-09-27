import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def initialize_weights(n_features):
    return np.random.randn(n_features)

def predict(weights, X):
    net_input = np.dot(X, weights)
    return np.where(net_input >= 0.0, 1, -1)

def train(X, y, eta=0.01, n_iter=10):
    n_samples, n_features = X.shape
    weights = initialize_weights(n_features)

    for _ in range(n_iter):
        errors = 0
        for xi, target in zip(X, y):
            update = eta * (target - predict(weights, xi)) 
            weights += update * xi
            errors += int(update != 0.0)

    return weights

'''linearly seperable data'''
# Define features X1 and X2 for two classes
class1_X1 = np.array([2.5, 1.5, 3.5, 3.0, 1.0])
class1_X2 = np.array([3.0, 2.0, 4.0, 3.5, 1.5])

class2_X1 = np.array([-2.0, -3.0, -2.5, -1.5, -1.0])
class2_X2 = np.array([-2.5, -3.5, -2.0, -1.0, -1.5])

# Create the dataset
X1 = np.concatenate((class1_X1, class2_X1))
X2 = np.concatenate((class1_X2, class2_X2))
X = np.column_stack((X1, X2))

# Create the target labels (1 for class 1, -1 for class 2)
y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])


perceptron_weights = train(X, y, eta=0.01, n_iter=10)

plt.scatter(class1_X1, class1_X2, label='Class 1', marker='o', c='blue')
plt.scatter(class2_X1, class2_X2, label='Class 2', marker='x', c='red')

# Plot the decision boundary
x_line = np.linspace(-4, 4, 100)
y_line = (-perceptron_weights[1] / perceptron_weights[0]) * x_line
plt.plot(x_line, y_line, '-g', label='Decision Boundary')

plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='best')
plt.title('Perceptron Decision Boundary')
plt.grid(True)
plt.show()

# Make predictions on the training set
predictions = predict(perceptron_weights, X)

# Calculate accuracy
accuracy = np.mean(predictions == y) * 100
print(f"Accuracy on the training set: {accuracy:.2f}%")

'''non linearly seperable data'''

# Define features X1 and X2 for two classes
class1_X1 = np.array([2.5, 1.5, 3.5, 3.0, 1.0])
class1_X2 = np.array([3.0, 2.0, 4.0, 3.5, 1.5])

class2_X1 = np.array([-2.0, -3.0, -2.5, -1.5, -1.0, 1.5, 0.5, 2.0, 2.5])
class2_X2 = np.array([-2.5, -3.5, -2.0, -1.0, -1.5, 0.5, 1.5, 1.0, 3.0])

# Create the dataset
X1 = np.concatenate((class1_X1, class2_X1))
X2 = np.concatenate((class1_X2, class2_X2))
X = np.column_stack((X1, X2))

# Create the target labels (1 for class 1, -1 for class 2)
y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

# Train the perceptron
perceptron_weights = train(X, y, eta=0.01, n_iter=100)

# Plot the number of errors in each epoch
plt.plot(range(1, len(perceptron_weights.errors_) + 1), perceptron_weights.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.title('Perceptron Convergence')
plt.grid(True)
plt.show()
