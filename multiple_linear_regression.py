# Import necessary libraries
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the diabetes dataset
diabetes = load_diabetes()

# Use all features (10 in total) for multiple linear regression
X = diabetes.data
y = diabetes.target

# Create the linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions on the training set
y_pred = model.predict(X)

# Calculate and print the Mean Squared Error
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Display coefficients for each feature
print("Feature coefficients:")
for feature, coef in zip(diabetes.feature_names, model.coef_):
    print(f"{feature}: {coef:.2f}")

# Plot the actual vs. predicted values
plt.scatter(y, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal fit')
plt.xlabel('Actual Disease Progression')
plt.ylabel('Predicted Disease Progression')
plt.title('Multiple Linear Regression: Diabetes Dataset')
plt.legend()
plt.show()
