import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
class AdalineSGD(object):
    """ADAptive LInear NEuron classifier with Stochastic Gradient Descent.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for reproducibility.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_initialized = False

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            X, y = self._shuffle(X, y)
            cost = []
            print(self.w_)

            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)



        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights."""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data."""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers."""
        self.rgen = np.random.default_rng(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights."""
        output = self.activation(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
        

    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation."""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.activation(X) >= 0, 1, -1)
    


# Load the Titanic dataset
data = pd.read_csv('train.csv')

# Preprocess the data with one-hot encoding for 'Sex'
def preprocess_data(df):
    selected_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']
    X = df[selected_features]
    y = df['Survived']
    
    #handle missing data
    X.loc[:, 'Age'].fillna(X['Age'].median(), inplace=True)

    # One-hot encode the 'Sex' column
    X['IsFemale'] = (X['Sex'] == 'female').astype(int)
    X.drop(columns=['Sex'], inplace=True)
    
    print(X)
    return X, y

X, y = preprocess_data(data)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Create an instance of AdalineSGD
adaline = AdalineSGD(eta=0.000001, n_iter=30, random_state=1)

# Train the model on the training data
adaline.fit(X_train.values, y_train.values)

# Evaluate the model on training data
train_predictions = adaline.predict(X_train.values)
train_accuracy = (train_predictions == y_train.values).mean()

# Evaluate the model on testing data
test_predictions = adaline.predict(X_test.values)
test_accuracy = (test_predictions == y_test.values).mean()

print(test_predictions)

print(f"Accuracy on training data: {train_accuracy * 100:.2f}%")
print(f"Accuracy on testing data: {test_accuracy * 100:.2f}%")


# Access the feature weights
feature_weights = adaline.w_[1:]
print(feature_weights)

# Normalize the feature weights
normalized_weights = (feature_weights - feature_weights.min()) / (feature_weights.max() - feature_weights.min())

# Create a DataFrame to store the feature names and their normalized weights
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Normalized Weight': normalized_weights})

# Sort the features by normalized weight in descending order to see the most predictive features
sorted_feature_importance = feature_importance_df.sort_values(by='Normalized Weight', ascending=False)

# Print or visualize the sorted feature importances
print(sorted_feature_importance)