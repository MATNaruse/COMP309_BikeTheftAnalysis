# -*- coding: utf-8 -*-
# External Imports
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix
import joblib

# Local Imports
import SupportFunctions as SF

"""
Data Exploration
"""

# Loading Data
bikedata = pd.read_csv('D:/School/Fall2020/COMP309_BikeTheftAnalysis/Dataset/Bicycle_Thefts.csv')
# Quick View of Data
print(bikedata.columns.values)
print(bikedata.info())

# Trimming irrelevant columns
bikedata = bikedata.drop(["X", "Y", "FID", "Index_", "event_unique_id"], axis=1)

# Sorting Categorical/String Varible Columns
categorical_columns = SF.get_cat_col(bikedata,"bikedata", True)

# Finding out which columns have missing values 
# *Note: Already 'cleaned' numerical values
SF.disp_col_w_missing(bikedata, "bikedata", categorical_columns)
# Expected Columns
# Bike_Model   8141
# Bike_Colour  1729

# As we are CURRENTLY not focusing on Bike_Model and Bike_Colour, don't need
#   to worry about them for now (?)


"""
Data Modeling
"""
# List of Features to focus on
FeatureSelection = ["Location_Type", "Premise_Type", "Status", "Hood_ID"]
FS_bikedata = bikedata[FeatureSelection]

# Getting Categorical Columns for Dummy Generation
FS_bikedata_cat_col = SF.get_cat_col(FS_bikedata, "FS_bikedata")
FS_bikedata_dumm = pd.get_dummies(FS_bikedata, columns=FS_bikedata_cat_col, dummy_na=False)
print("\nConfirming Missing Data(?):\n===========================")
print(len(FS_bikedata_dumm) - FS_bikedata_dumm.count())

# Creating Scalar Object
scaler = preprocessing.StandardScaler()
scaled_bikedata = scaler.fit_transform(FS_bikedata_dumm)
scaled_bikedata = pd.DataFrame(scaled_bikedata, columns=FS_bikedata_dumm.columns)
scaled_bikedata.describe()


"""
Predictive Model Building
"""
# Outcome Column
dependent_variable = 'Status_RECOVERED'

# DEBUG - Actual Recovery Numbers
print("\nActual Recovery Numbers")
print("=======================")
print(FS_bikedata_dumm['Status_RECOVERED'].value_counts())

# Splitting Data for Train/Test
x = scaled_bikedata[scaled_bikedata.columns.difference([dependent_variable])]
y = scaled_bikedata[dependent_variable]
y = y.astype(int)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

# Building the Model
lr = LogisticRegression(solver="lbfgs")
lr.fit(x,y)


"""
Model Scoring & Evaluation
"""

# KFold and Cross Validation
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
score = np.mean(cross_val_score(lr, xTrain, yTrain, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print(f"\nKFold 10 Score: [{score}]")

# Accuracy
yTest_predict = lr.predict(xTest)
print("\nAccuracy:", metrics.accuracy_score(yTest, yTest_predict))

# Confusion Matrix
labels = y.unique() # [0, 9]
print("\nConfusion Matrix")
print('================')
print(confusion_matrix(yTest, yTest_predict, labels=labels))


"""
Model Dumping
"""

joblib.dump(lr, "model_lr.pkl")
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')

print("Model and Model Columns Dumped!")

print("\nModel Columns:")
for col in model_columns:
    print(f"\t- {col}")