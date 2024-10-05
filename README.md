# Linear Regression Example:

This project demonstrates a simple implementation of Linear Regression using Python's scikit-learn library and a sample dataset. The goal of this example is to show how to build a basic linear regression model using a single feature from the dataset and plot the results for better understanding.

Features:

Linear Regression: A model that fits a linear equation to the observed data.
Visualization: A scatter plot showing the actual data points and the fitted regression line to visualize the relationship between the input feature and the target variable.
Requirements
To run this project, you'll need the following Python libraries installed:

```python
scikit-learn
matplotlib
numpy
```

You can install the required libraries using the following command:

bash
Copy code

```python
pip install scikit-learn matplotlib numpy
```

Running the Code
Clone the repository to your local machine (after uploading it to GitHub).
Ensure all dependencies are installed.
Run the Python script:
bash
Copy code

```python
python linear_regression.py
```

This will create a plot showing the actual data points (in blue) and the fitted regression line (in red).

Understanding the Code:

Loading the Dataset: The dataset is loaded using `load_diabetes()` from scikit-learn.
Feature Selection: Only a single feature (e.g., BMI) is used for this example.
Model Training: A LinearRegression model is created and trained on the selected feature and the target variable.
Prediction and Visualization: The model predicts the target variable, and the results are plotted using matplotlib.

Example Screenshot if running is successful:


![image](https://github.com/user-attachments/assets/3413d944-05db-464e-9df2-2a8f15068589)

## Extension to Linear Regression Example (Using Multiple Features):

This updated version of the Linear Regression Example project adds flexibility by allowing users to choose from multiple features in the dataset. Users can input real-world values, and the code will normalize these values and make predictions accordingly. The results are visualized using matplotlib.

Code Breakdown:

A list of features is created with their real-world descriptions and typical ranges (e.g., BMI, cholesterol, blood pressure).
The program displays these features, prompting the user to choose one by index. Each feature has an associated real-world range for input.
Real-World Input:

After the user selects a feature, they are asked to input a real-world value for that feature.
This value is then normalized using min-max normalization to match the normalized data in the dataset. This step is crucial because the dataset values are already normalized, but the user is providing real-world data.
 
This scales the input into the correct range that matches the dataset.

Linear Regression:

Once the input is normalized, a Linear Regression model is created using the selected feature.
The model is then trained on the dataset, and predictions are made for both the entire dataset and the specific input provided by the user.
Visualization:

A scatter plot is created showing the actual data points (blue dots) and the predicted regression line (red line) for the selected feature.
The user’s input and the corresponding prediction are shown as a red dot on the plot.
User-Friendly Outputs:

The predicted value for the given real-world input is printed, along with a visualization of the linear regression fit.
Input and Output Example:
Input:
Suppose the user selects BMI as the feature.
They input a BMI value of 18 (within the typical real-world range of 15–40 for BMI).
Output:
The model predicts the corresponding disease progression based on the input BMI value.
The results are shown as follows:

A scatter plot of BMI vs. disease progression.
A red regression line showing the model’s predictions.
A red dot on the regression line showing the prediction for BMI = 18.

Input:
```python
Available features with descriptions and typical real-world ranges:
0: Age (typical range: 20 to 80)
1: Sex (typical range: -1 to 1)
2: BMI (Body Mass Index, typical range: 15 to 40)
3: Blood Pressure (typical range: 80 to 180)
...
Select a feature by entering its index (0-9): 2
Enter a value for BMI (Body Mass Index): 18
```

Example Screenshot if running is successful:

![image](https://github.com/user-attachments/assets/0dabb3d0-2865-4008-a970-c6c7c843b9f9)
