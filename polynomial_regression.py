# Import necessary libraries
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np

# Load the diabetes dataset
#Different data sets can be loaded
diabetes = load_diabetes()

# Use only one feature (e.g., BMI feature at index 2)
X = diabetes.data[:, np.newaxis, 2]  # Reshaping to make it 2D
y = diabetes.target

# Set the degree of the polynomial
degree = 2  # You can experiment with 3 or 4 as well

# Create a pipeline with PolynomialFeatures and LinearRegression
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Fit the model to the data
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label=f'Polynomial Regression (degree {degree})')
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title('Polynomial Regression: Diabetes Dataset')
plt.legend()
plt.show()
