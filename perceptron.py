import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

def initialize_weights(n_features):
    return np.random.randn(n_features)

def predict(weights, X):
    net_input = np.dot(X, weights)
    return np.where(net_input >= 0.0, 1, -1)

def train(X, y, eta=0.01, n_iter=10):
    n_samples, n_features = X.shape
    weights = initialize_weights(n_features)
    errors = []

    for _ in range(n_iter):
        error = 0
        for xi, target in zip(X, y):
            update = eta * (target - predict(weights, xi)) 
            weights += update * xi
            error += int(update != 0.0)
        errors.append(error)
    return weights, errors

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


perceptron_weights, error_count = train(X, y, eta=0.01, n_iter=10)


x_line = np.linspace(-4, 4, 100)
y_line = (-perceptron_weights[1] / perceptron_weights[0]) * x_line
plt.scatter(class1_X1, class1_X2, label='Class 1', marker='o', c='blue')
plt.scatter(class2_X1, class2_X2, label='Class 2', marker='x', c='red')
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
print(f"Accuracy on the training set(Linearly Seperable): {accuracy:.2f}%")

plt.plot(range(1, len(error_count) + 1), error_count, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.title('Perceptron Convergence')
plt.grid(True)
plt.show()

'''test using random weights'''
random_weights = [-0.3123, 0.21343]

def predict_random_weights(X, weights):
    net_input = np.dot(X, weights) + weights[0]
    return np.where(net_input >= 0, 1, -1)


random_predictions = predict_random_weights(X, random_weights)
random_accuracy = np.mean(random_predictions == y) * 100
print(f"Random Baseline Model Accuracy: {random_accuracy:.2f}%")


'''non linearly seperable data'''

# Define features X1 and X2 for two classes
class1_X1 = np.array([2.5, 1.5, 3.5, 3.0, 1.0, -1.0, -2.0, -2.5, -3.0, -3.5])
class1_X2 = np.array([3.0, 2.0, 4.0, 3.5, 1.5, -1.5, -1.0, -2.0, -3.0, -2.5])

class2_X1 = np.array([0.5, 0.0, -0.5, -1.0, -1.5, 1.5, 1.0, 2.0, 2.5, 3.0])
class2_X2 = np.array([0.5, 0.0, -0.5, -1.0, -1.5, 1.5, 1.0, 2.0, 2.5, 3.0])

# Create the dataset
X1 = np.concatenate((class1_X1, class2_X1))
X2 = np.concatenate((class1_X2, class2_X2))
X = np.column_stack((X1, X2))

# Create the target labels (1 for class 1, -1 for class 2)
y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

# Train the perceptron
perceptron_weights, error_count = train(X, y, eta=0.001, n_iter=30)

# Plot the number of errors in each epoch
plt.plot(range(1, len(error_count) + 1), error_count, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.title('Perceptron Convergence')
plt.grid(True)
plt.show()


predictions = predict(perceptron_weights, X)

# Calculate accuracy
accuracy = np.mean(predictions == y) * 100
print(f"Accuracy on the training set(Non-Linearly Seperable): {accuracy:.2f}%")



'''Make predictions using random weights'''

random_weights = [2.313425, -3.13434 ]



random_predictions = predict_random_weights(X, random_weights)
random_accuracy = np.mean(random_predictions == y) * 100
print(f"Random Baseline Model Accuracy: {random_accuracy:.2f}%")