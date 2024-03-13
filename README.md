## Logistic Regression Model for Binary Classification

### Overview
This repository contains an implementation of logistic regression for binary classification using the scikit-learn library in Python. The model is trained on a dataset imported from a CSV file and then tested for classification accuracy.

### Contents
- `logistic_regression.py`: Python script containing the implementation of logistic regression.
- `Social_Network_Ads.csv`: CSV file containing the dataset used for training and testing the model.

### Requirements
- Python 3.x
- NumPy
- Matplotlib
- Pandas
- scikit-learn

### Usage
1. Clone the repository:

    ```
    git clone https://github.com/yourusername/Logistic-Regression-Model.git
    ```

2. Navigate to the repository:

    ```
    cd Logistic-Regression-Model
    ```

3. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

4. Run the `logistic_regression.py` script:

    ```
    python logistic_regression.py
    ```

### Detailed Description
1. The necessary libraries are imported, including numpy, matplotlib.pyplot, and pandas.
2. The dataset is imported from a CSV file called 'Social_Network_Ads.csv' using pandas.
3. The dataset is divided into features (X) and the target variable (y).
4. The dataset is split into training and test sets using the train_test_split function from sklearn.model_selection.
5. Feature scaling is applied to standardize the features using the StandardScaler from sklearn.preprocessing.
6. A logistic regression model is created using the LogisticRegression class from sklearn.linear_model and trained on the training set.
7. A new prediction is made on a sample input ([30, 87000]) using the trained classifier.
8. Predictions are made on the test set, and the predicted and actual values are printed.
9. The confusion matrix and accuracy score are calculated using the confusion_matrix and accuracy_score functions from sklearn.metrics.
10. The training set results are visualized using a contour plot, displaying the decision boundary of the logistic regression model.
11. The test set results are visualized in a similar manner to evaluate the model's performance.

### Note
- Ensure that the CSV file 'Social_Network_Ads.csv' exists and contains the appropriate data. The dataset should have two numerical columns (age and estimated salary) and a binary target variable.


