import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from adalineSGD import AdalineSGD

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