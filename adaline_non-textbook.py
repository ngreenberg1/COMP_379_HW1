import numpy as np
import pandas as pd

def random_weights(X, random_state: int):
    '''create vector of random weights
    Parameters
    ----------
    X: 2-dimensional array, shape = [n_samples, n_features]
    Returns
    -------
    w: array, shape = [w_bias + n_features]'''
    rand = np.random.RandomState(random_state)
    w = rand.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
    return w

def net_input(X, w):
    '''Compute net input as dot product'''
    return np.dot(X, w[1:]) + w[0]

def predict(X, w):
    '''Return class label after unit step'''
    return np.where(net_input(X, w) >= 0.0, 1, 0)

def fit(X, y, eta=0.001, n_iter=1):
    '''loop over exemplars and update weights'''
    mse_iteration = []
    w = random_weights(X, random_state=1)
    for pair in range(n_iter):
        output = net_input(X, w)
        gradient = 2*(y - output)
        w[1:] += eta*(X.T @ gradient)
        w[0] += eta*gradient.sum()
        mse = (((y - output)**2).sum())/len(y)
        mse_iteration.append(mse)
    return w, mse_iteration

df = pd.read_csv('train.csv')



df_shuffle = df.sample(frac=1, random_state=1).reset_index(drop=True)
X = df_shuffle[['Pclass', 'Age', 'Parch', 'Fare']]
y = df_shuffle['Survived'].to_numpy()



w, mse = fit(X, y, eta=0.001, n_iter=20)
y_pred = predict(X, w)
num_correct_predictions = (y_pred == y).sum()
accuracy = (num_correct_predictions / y.shape[0]) * 100
print('ADALINE accuracy: %.2f%%' % accuracy)
print(w)
print(y_pred)
