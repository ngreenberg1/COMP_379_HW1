import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
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

    def fit(self, X, y, lambda_=0.01):
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
        self.w_ = np.random.randn(1 + X.shape[1])
        self.cost_ = []

        cost = 0.0 

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)

            print(f"Iteration {i + 1}: Cost = {cost}") #for debugging purposes
            print(f"Iteration {i + 1}: weights = {self.w_}")

            self.w_[1:] += self.eta * (X.T.dot(errors) - lambda_ * self.w_[1:])  
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0+ lambda_ * (self.w_[1:]**2).sum() 
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
        return np.where(self.activation(X) >= 0.0, 1, 0)
  
data = pd.read_csv('train.csv')

def preprocess_data(df):

    # Select relevant features and target variable

    # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

    selected_features = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare']

    X = df[selected_features]

    y = df['Survived']

 

    # Handle missing values (you can choose an appropriate strategy)

    X['Age'].fillna(X['Age'].median(), inplace=True)

    label_encoder = LabelEncoder()

    # Encode categorical variables (e.g., 'Sex' and 'Embarked'

    # One-hot encode the 'Sex' column
    X['IsFemale'] = (X['Sex'] == 'female').astype(int)
    X.drop(columns=['Sex'], inplace=True)
    
    print(X)
    

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y



X, y = preprocess_data(data)




# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


adaline = AdalineGD(eta=0.009, n_iter=50)    #eta 0.009, n_iter 20
adaline.fit(X_train, y_train, lambda_=0.01)

y_train_pred = adaline.predict(X_train)
y_test_pred = adaline.predict(X_test)

accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print(f"Accuracy on training data: {accuracy_train * 100:.2f}%")
print(f"Accuracy on test data: {accuracy_test * 100:.2f}%")


print(y_test_pred)



