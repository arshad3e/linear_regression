# multiple linear regression where various features can be included to get prediction
# Import necessary libraries
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

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

# Set up cross-validation with Mean Squared Error as the scoring metric
# Here we use 5-fold cross-validation
mse_scorer = make_scorer(mean_squared_error)
cv_scores = cross_val_score(model, X, y, cv=5, scoring=mse_scorer)

# Print cross-validation scores (MSE for each fold)
print("Cross-validation Mean Squared Error (MSE) for each fold:")
for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score:.2f}")

# Calculate and print the average MSE across all folds
mean_cv_mse = np.mean(cv_scores)
print(f"\nAverage Cross-Validation MSE: {mean_cv_mse:.2f}")

# Fit the model on the entire dataset and calculate MSE without cross-validation
model.fit(X, y)
y_pred = model.predict(X)
train_mse = mean_squared_error(y, y_pred)
print(f"\nTraining MSE (no cross-validation): {train_mse:.2f}")
