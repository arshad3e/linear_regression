# Import necessary libraries
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Load the diabetes dataset
diabetes = load_diabetes()

# Use only one feature (e.g., BMI feature at index 2)
X = diabetes.data[:, np.newaxis, 2]  # Reshaping to make it 2D
y = diabetes.target

# Create the linear regression model
model = LinearRegression()

# Fit the model with the data
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Fitted Line')
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title('Linear Regression: Diabetes Dataset')
plt.legend()
plt.show()
