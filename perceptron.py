import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def step_function(x):
    return np.where(x > 0, 1, -1)

class Perceptron(object):
    
    #constructor
    def __init__(self, learning_rate=0.01, n_iter=20) :
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.activation_function = step_function
        self.weights = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize parameters
        self.weights = np.random.randn(n_features)
        self.errors_ = []

        y_ = np.where(y > 0, 1, -1)

        for _ in range(self.n_iter):
            
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights)
                y_predicted = self.activation_function(linear_output)

                #perceptron update rule
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i

               
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.weights)
        y_predicted = self.activation_function(linear_output)
        return y_predicted
    


# Testing
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

df = pd.read_csv('linearlyseperable.csv')
print(df.to_string())
print(df.columns)


df['Classlabel'] = np.where(df['Classlabel'] == 'Adult', -1, 1)

# Select 'Height' and 'Weight' as features (X) and 'Classlabel' as labels (y)
X = df[['Height', 'Weight']].values
y = df['Classlabel'].values

# Create a scatter plot for 'Adult' data points (red circles)
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', marker='o', label='Adult')

# Create a scatter plot for 'Child' data points (blue crosses)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='x', label='Child')

# Set labels for the x and y axes
plt.xlabel('Height')
plt.ylabel('Weight')

# Add a legend to the plot in the upper-left corner
plt.legend(loc='upper left')

# Display the scatter plot
plt.show()

perceptron = Perceptron(learning_rate=0.0001, n_iter=10)

perceptron.fit(X, y)

predictions = perceptron.predict(X)

plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

x0_1 = np.amin(X[:, 0])
x0_2 = np.amax(X[:, 0])

x1_1 = (-perceptron.weights[0] * x0_1) / perceptron.weights[1]
x1_2 = (-perceptron.weights[0] * x0_2) / perceptron.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

ymin = np.amin(X[:, 1])
ymax = np.amax(X[:, 1])
ax.set_ylim([ymin - 3, ymax + 3])

plt.show()