# -*- coding: utf-8 -*-
# External Imports
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz
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
Predictors = ["Location_Type", "Premise_Type", "Status", "Division"]
Pred_bikedata = bikedata[Predictors]
Target = 'Status_RECOVERED'


# # Prepping 'Occurrence_Time'
# occTime_parsed = SF.parse_time(FS_bikedata['Occurrence_Time'])
# FS_bikedata['Occurrence_Time'] = occTime_parsed

# Getting Categorical Columns for Dummy Generation
Pred_bikedata_cat_col = SF.get_cat_col(Pred_bikedata, "Pred_bikedata")
Pred_bikedata_dumm = pd.get_dummies(Pred_bikedata, columns=Pred_bikedata_cat_col, dummy_na=False)
Predictors = Pred_bikedata_dumm.columns.values
print("\nConfirming Missing Data(?):\n===========================")
print(len(Pred_bikedata_dumm) - Pred_bikedata_dumm.count())

"""
Predictive Model Building
"""
# Outcome Column

# DEBUG - Actual Recovery Numbers
print("\nActual Recovery Numbers")
print("=======================")
print(Pred_bikedata_dumm['Status_RECOVERED'].value_counts())
# Actual Recovery Numbers
# =======================
# 0    21332 -> STOLEN or UNKNOWN
# 1      252 -> Actually RECOVERED

Pred_bikedata_dumm['is_train'] = np.random.uniform(0, 1, len(Pred_bikedata_dumm)) <= .75

train, test = Pred_bikedata_dumm[Pred_bikedata_dumm['is_train']==True], Pred_bikedata_dumm[Pred_bikedata_dumm['is_train']==False]

print('Number of observations in the training data:', len(train))

print('Number of observations in the test data:',len(test))

dTree = DecisionTreeClassifier(criterion='entropy', min_samples_split=20, random_state=99)

dTree.fit(train[Predictors], train[Target])

preds = dTree.predict(test[Predictors])

pd.crosstab(test['Status_RECOVERED'], preds, rownames=['Actual'], colnames=['Predictions'])


x = Pred_bikedata_dumm[Predictors]
y = Pred_bikedata_dumm[Target]

trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)

score = np.mean(cross_val_score(dTree, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print(score) 


testY_predict = dTree.predict(testX)

labels = y.unique()

print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=labels))

# I have no clue
# cm = confusion_matrix(testY, testY_predict, labels)

# ax = plt.subplot()

# sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');

# ax.set_title('Confusion Matrix');

# ax.xaxis.set_ticklabels(['Stolen/Unknown','Recovered']); 
# ax.yaxis.set_ticklabels(['Stolen/Unknown','Recovered']);


# """
# Model Scoring & Evaluation
# """


# """
# Model Dumping
# """

joblib.dump(dTree, "model_dTree.pkl")
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')

print("Model and Model Columns Dumped!")

print("\nModel Columns:")
for col in model_columns:
    print(f"\t- {col}")