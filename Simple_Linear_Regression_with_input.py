# Import necessary libraries
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Load the diabetes dataset
diabetes = load_diabetes()

# Updated feature names with descriptions and typical ranges
feature_info = [
    {"name": "Age", "range": (20, 80)},                     # Age range in years
    {"name": "Sex", "range": (-1, 1)},                      # Sex (1: Male, -1: Female)
    {"name": "BMI (Body Mass Index)", "range": (15, 40)},   # BMI in real-world values
    {"name": "Blood Pressure", "range": (80, 180)},         # Blood pressure in mm Hg
    {"name": "Total Cholesterol", "range": (100, 300)},     # Cholesterol in mg/dL
    {"name": "LDL Cholesterol", "range": (70, 200)},        # LDL Cholesterol in mg/dL
    {"name": "HDL Cholesterol", "range": (30, 100)},        # HDL Cholesterol in mg/dL
    {"name": "Triglycerides/Free Fatty Acids", "range": (50, 250)},  # Triglycerides in mg/dL
    {"name": "Serum Glucose", "range": (70, 200)},          # Glucose in mg/dL
    {"name": "Insulin or Inflammatory Marker", "range": (2, 50)}  # Insulin in μU/mL
]

# Display the features and their real-world ranges
print("Available features with descriptions and typical real-world ranges:")
for i, feature in enumerate(feature_info):
    print(f"{i}: {feature['name']} (typical range: {feature['range'][0]} to {feature['range'][1]})")

# Ask the user to select a feature by index
feature_index = int(input(f"Select a feature by entering its index (0-{len(feature_info)-1}): "))

# Ask for the real-world input value for the selected feature
input_value = float(input(f"Enter a value for {feature_info[feature_index]['name']}: "))

# Get the min and max values for the selected feature
min_value, max_value = feature_info[feature_index]["range"]

# Normalize the input value (scale it to match the normalized dataset range)
# The formula for min-max normalization is: X_scaled = (X_real - min) / (max - min)
normalized_input_value = (input_value - min_value) / (max_value - min_value)

# Use the selected feature for linear regression
X = diabetes.data[:, np.newaxis, feature_index]  # Reshaping to make it 2D
y = diabetes.target

# Create the linear regression model
model = LinearRegression()

# Fit the model with the data
model.fit(X, y)

# Make predictions for the entire dataset
y_pred = model.predict(X)

# Predict the disease progression for the normalized input value
predicted_progression = model.predict([[normalized_input_value]])

# Print the predicted value
print(f"Predicted disease progression for {feature_info[feature_index]['name']} value {input_value}: {predicted_progression[0]}")

# Plot the actual data points (selected feature vs. disease progression)
plt.scatter(X, y, color='blue', label='Actual Data')

# Plot the regression line (selected feature vs. predicted disease progression)
plt.plot(X, y_pred, color='red', label='Fitted Line')

# Plot the input feature value and predicted disease progression in red
plt.scatter([normalized_input_value], predicted_progression, color='red', marker='o', s=100, label=f'Input {feature_info[feature_index]["name"]} ({input_value}) Prediction')

# Add labels and title to the plot
plt.xlabel(feature_info[feature_index]["name"])
plt.ylabel('Disease Progression')
plt.title(f'Linear Regression: {feature_info[feature_index]["name"]} vs. Disease Progression')

# Add legend
plt.legend()

# Show the plot
plt.show()
