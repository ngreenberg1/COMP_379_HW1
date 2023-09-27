import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.linear_model import SGDClassifier
from numpy.random import seed


class AdalineGD(object):
    """ADAptive LInear NEuron classifier.
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    """
    def __init__(self, eta=0.001, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors,
            where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)

            print("Data type of errors:", errors.dtype) #for debugging purposes
            print("Data type of self.w_:", self.w_.dtype) #for debugging purposes

            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
        
        
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
  


titanic_df = pd.read_csv('train.csv')
titanic_df = titanic_df.drop(columns=['PassengerId', 'Cabin', 'Name', 'Ticket'])
titanic_df = pd.get_dummies(titanic_df, columns=['Sex'], drop_first=True)
titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'])
print(titanic_df.head)

X = titanic_df[['Pclass', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].values
y = titanic_df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0 )



adaline = AdalineGD(eta=0.0001, n_iter=10)

adaline.fit(X_train, y_train)

y_train_pred = adaline.predict(X_train)
y_test_pred = adaline.predict(X_test)

accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print(f"Accuracy on training data: {accuracy_train * 100:.2f}%")
print(f"Accuracy on test data: {accuracy_test * 100:.2f}%")




