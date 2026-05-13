# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load & Explore Dataset — Read the Employee CSV file and inspect the data using head(), info(), null checks, and target class distribution.

2.Preprocess Data — Apply Label Encoding on the categorical "salary" column to convert it into numerical format.

3.Split Data — Separate features (satisfaction level, evaluation, projects, hours, etc.) and target ("left"), then split into 80% training and 20% testing sets.

4.Train & Predict — Build a Decision Tree Classifier using entropy criterion, fit it on training data, and predict outcomes on test data.

5.Evaluate & Visualize — Calculate accuracy score, predict on a sample input, and plot the decision tree for visual interpretation.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ishwarya s
RegisterNumber:  212225240053
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

file_path = 'Employee.csv'
data = pd.read_csv(file_path)

data = pd.get_dummies(data)

X = data.drop('left', axis=1)
y = data['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['Stayed', 'Left'], filled=True)
plt.show()
```

## Output:
<img width="1262" height="658" alt="Screenshot 2026-05-13 104730" src="https://github.com/user-attachments/assets/41e69861-deee-4d3e-a40d-837b2589a3e9" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
