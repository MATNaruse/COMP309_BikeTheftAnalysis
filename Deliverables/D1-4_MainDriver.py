# -*- coding: utf-8 -*-
# External Imports
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing, metrics
from sklearn.utils import resample
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
bikedata = pd.read_csv(os.path.join(Path(__file__).parents[1],
                                        "Dataset\Bicycle_Thefts.csv"))
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
FeatureSelection = ["Location_Type", "Premise_Type", "Status", "Neighbourhood"]
FS_bikedata = bikedata[FeatureSelection]

    
# Getting Categorical Columns for Dummy Generation
FS_bikedata_cat_col = SF.get_cat_col(FS_bikedata, "FS_bikedata")
FS_bikedata_dumm = pd.get_dummies(FS_bikedata, columns=FS_bikedata_cat_col, dummy_na=False)
print("\nConfirming Missing Data(?):\n===========================")
print(len(FS_bikedata_dumm) - FS_bikedata_dumm.count())

# # Creating Scalar Object
# scaler = preprocessing.StandardScaler()
# scaled_bikedata = scaler.fit_transform(FS_bikedata_dumm)
# scaled_bikedata = pd.DataFrame(scaled_bikedata, columns=FS_bikedata_dumm.columns)
# scaled_bikedata.describe()

scaled_bikedata = FS_bikedata_dumm

# Outcome Column
dependent_variable = 'Status_RECOVERED'

# Splitting Data for Train/Test
x = scaled_bikedata[scaled_bikedata.columns.difference([dependent_variable])]
y = scaled_bikedata[dependent_variable]
y = y.astype(int)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)

# Managing Imbalanced Classes
# Down-sizing number of STOLEN or UNKNOWN
bd_majority = FS_bikedata[FS_bikedata.Status!="RECOVERED"]
bd_minority = FS_bikedata[FS_bikedata.Status=="RECOVERED"]

bd_majority_downsampled= resample(bd_majority, replace=False, n_samples=len(bd_minority), random_state=123)

bd_downsampled = pd.concat([bd_majority_downsampled, bd_minority])

print(bd_downsampled.Status.value_counts())


"""
Predictive Model Building
"""
# DEBUG - Actual Recovery Numbers
print("\nActual Recovery Numbers")
print("=======================")
print(FS_bikedata_dumm['Status_RECOVERED'].value_counts())
# Actual Recovery Numbers
# =======================
# 0    21332 -> STOLEN or UNKNOWN
# 1      252 -> Actually RECOVERED


# Building the Model
lr = LogisticRegression(solver="lbfgs")
lr.fit(xTrain,yTrain)


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