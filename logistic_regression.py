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

# Plot the Histogram of Predicted Churn Probabilities
threshold = 0.5 #This is the cutoff point we will use to classify customers as likely to churn (1) or not (0).

plt.figure(figsize=(8, 5))
plt.hist(predictions, bins=50, color='darkblue', edgecolor='black')
plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
plt.title('Predicted Churn Probabilities')
plt.xlabel('Probability of Churn')
plt.ylabel('Count')
plt.legend()  
plt.tight_layout()
plt.show()

#Pulls the coefficients from the trained logistic regression model 
# and identifies the top 15 features that have the most impact on predicting customer churn. 
# It then creates a horizontal bar chart to visualize these features, coloring the bars red for 
# features that increase churn risk and blue for those that decrease it.

# Get the coefficient for each feature
# Positive = increases churn risk, Negative = decreases churn risk
coefficients = pd.Series(model.coef_[0], index=X_train.columns)

# Sort by absolute value to find the most impactful features
top15 = coefficients.abs().nlargest(15).index

# Get the actual coefficient values for those top 15 features
top15_coefs = coefficients[top15].sort_values()

# Color the bars based on whether they increase or decrease churn
#Red = increases churn risk (positive coefficient)
#Blue = decreases churn risk (negative coefficient)
bar_colors = []
for value in top15_coefs:
    if value > 0:
        bar_colors.append('maroon')   # increases churn risk
    else:
        bar_colors.append('darkblue') # decreases churn risk

# Plot the top 15 features with a horizontal bar chart
plt.figure(figsize=(9, 6))
plt.barh(top15_coefs.index, top15_coefs.values, color=bar_colors, edgecolor='black')
plt.axvline(x=0, color='black', linestyle='--')
plt.title('Top 15 Features That Influence Churn')
plt.xlabel('Coefficient Value (red = more churn, blue = less churn)')
plt.tight_layout()
plt.show()

# This section calculates the churn rate for each category in the dataset and visualizes it with a bar chart.

numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges'] # These are the numeric columns in the dataset that we will exclude when looking for category columns.

# Find all the category columns (they only contain 0s and 1s)
category_columns = []
for col in X_train.columns:
    if col not in numeric_columns and X_train[col].nunique() == 2:
        category_columns.append(col)

# For each category, calculate the churn rate of customers in that category
churn_rates = {}
for col in category_columns:
    # Find all customers who belong to this category
    customers_in_category = X_train[col] == 1

    # Skip if no customers belong to this category
    if customers_in_category.sum() == 0:
        continue
    # Calculate what percentage of those customers churned
    churn_rate = y_train[customers_in_category].mean() * 100

# Put results into a dataframe and sort highest to lowest
churn_df = pd.DataFrame({
    'Category': list(churn_rates.keys()),
    'ChurnRate': list(churn_rates.values())
})
churn_df = churn_df.sort_values('ChurnRate', ascending=False).reset_index(drop=True)

# Calculate the overall churn rate to use as a reference line
overall_churn_rate = y_train.mean() * 100

# Colour each bar based on whether the churn rate is above or below the overall churn rate
#Red = higher churn than average, Blue = lower churn than average

bar_colors2 = []
for rate in churn_df['ChurnRate']:
    if rate > overall_churn_rate:
        bar_colors2.append('Maroon')   # higher churn than average
    else:
        bar_colors2.append('darkblue') # lower churn than average

# Plot the churn rate for each category with a bar chart, including a dashed line for the overall churn rate.
plt.figure(figsize=(12, 7))
plt.bar(churn_df['Category'], churn_df['ChurnRate'], color=bar_colors2, edgecolor='black')
plt.axhline(y=overall_churn_rate, color='black', linestyle='--',
            label=f'Overall Churn Rate ({overall_churn_rate:.1f}%)')
plt.title('Churn Rate by Category')
plt.ylabel('Churn Rate (%)')
plt.xlabel('Category')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.legend()
plt.tight_layout()
plt.show()
