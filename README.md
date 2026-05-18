# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas 
2.Import Decision tree classifier
3.Fit the data in the model
4.Find the accuracy score 
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MONISHA.A.K
RegisterNumber: 212225230187 
*/
```import pandas as pd
from google.colab import files
import io

# Upload CSV file
print("Please choose Employee.csv file")

uploaded = files.upload()

# Get uploaded file name
file_name = list(uploaded.keys())[0]

# Read CSV file
data = pd.read_csv(io.BytesIO(uploaded[file_name]))

# Display first rows
print("\nData Head:")
print(data.head())

# Dataset information
print("\nData Info:")
print(data.info())

# Check null values
print("\nNull Values:")
print(data.isnull().sum())

# Count values
print("\nLeft Value Counts:")
print(data["left"].value_counts())

# Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Encode salary column
data["salary"] = le.fit_transform(data["salary"])

print("\nEncoded Data Head:")
print(data.head())

# Feature selection
print("\nX Head:")

x = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary"
]]

print(x.head())

# Target variable
y = data["left"]

# Split dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)

# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="entropy")

# Train model
dt.fit(x_train, y_train)

# Prediction
y_pred = dt.predict(x_test)

# Accuracy
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)

print("\nAccuracy Value:")
print(accuracy)

# Single prediction
print("\nData Prediction:")

prediction = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])

print(prediction)

# Plot Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))

plot_tree(
    dt,
    feature_names=x.columns,
    class_names=['Not Left', 'Left'],
    filled=True
)

plt.show()
```

## Output:
<img width="1429" height="873" alt="Screenshot 2026-05-18 161033" src="https://github.com/user-attachments/assets/5523a37f-b62b-4ebf-9e64-4904250ac47e" />
<img width="1176" height="786" alt="Screenshot 2026-05-18 161044" src="https://github.com/user-attachments/assets/115b76d2-8046-4c9f-8951-eecb50e00ca9" />
<img width="1180" height="777" alt="image" src="https://github.com/user-attachments/assets/aa94c7e9-51bd-4a8f-8437-6d649805c602" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
