import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load the modified datasets 
#These files were created in the preprocessing step where we cleaned and encoded the raw data. 
trainData = pd.read_csv('train_processed.csv')
testData = pd.read_csv('test_processed.csv')

X_train = trainData.drop('Churn', axis=1)   #contains all the columns the model will learn from, except the target variable 'Churn'
y_train = trainData['Churn'] #contains the target variable 'Churn' which indicates whether a customer churned (1) or not (0)

# Train the logistic regression model
#max_iter = 1000 to give the model enough iterations to find the best coefficients for all the features.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict_proba(testData)[:, 1] #predict_proba gives the probability of each class (churn or not churn) for each customer in the test set.
print(predictions)

