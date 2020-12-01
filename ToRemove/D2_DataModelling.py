# -*- coding: utf-8 -*-
"""
COMP309 Bike Theft Analysis
301 063 251 : Arthur Batista
300 549 638 : Matthew Naruse
301 041 132 : Trent B Minia
300 982 276 : Simon Ducuara
### ### ### : Full Name

1) Data Modelling:
    a) Data transformations – includes handling missing data, categorical 
        data management, data normalization and standardizations as needed.
        
    b) Feature selection – use pandas and sci-kit learn.

    c) Train, Test data spliting – use numpy, sci-kit learn.
    
    d) Managing imbalanced classes if needed. 
        Check here for info: https://elitedatascience.com/imbalanced-classes

"""

# External Imports
import os, pandas
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Local Imports
from D1_BikeData import BikeData


""" ==========================================================================
    PREPARING DATA
==========================================================================="""

BikeData = BikeData()

raw_dataset = BikeData.get_raw_dataset()
cleaned_ds = BikeData.get_clean_dataset()

"""===========================================================================
    PART A) Data transformations – includes handling missing data, 
            categorical data management, data normalization and 
            standardizations as needed.
==========================================================================="""

# Part A Start

trimmed_ds = BikeData.get_trimmed_dataset()

# Part A End

"""===========================================================================
    PART B) Feature selection – use pandas and sci-kit learn.
==========================================================================="""

# Part B Start

def val_to_num(lValues:[], reverse:bool = False) -> {}:
    """
    Generate a dict from List
    
    Parameters:
    -----------
    lValues:
        List of Values
    
    reverse:
        Use Values as Keys instead of Index

    Returns:
    --------
    out_dict:
        A Dictionary of input values
    """
    out_dict = {}
    # indexes = []
    # for i in range(0, len(lValues)):
    #     indexes.append(np.random.seed(1))

    # for new_idx, value in zip(indexes, list(lValues)):
    #     if(reverse):
    #         out_dict[value] = new_idx
    #     else:
    #         out_dict[new_idx] = value
        
    for value in list(lValues):
        if(reverse):
            out_dict[value] = list(lValues).index(value)
        else:
            out_dict[list(lValues).index(value)] = value
    return out_dict


# Stating which columns we're considering Features
FeatureSelection = ["Location_Type", "Occurrence_Time", "Hood_ID", "Status"]

# Creating Value to Number references for Categorical Values
loctypeToNum = val_to_num(cleaned_ds['Location_Type'].unique(), True)
statusToNum = val_to_num(cleaned_ds['Status'].unique(), True)
timeToNum = val_to_num(cleaned_ds['Occurrence_Time'].unique(), True)
# Creating a new DataFrame based on the Feature Selection
featSelc_dataset = cleaned_ds[FeatureSelection]
featSele_dataset_Cat = cleaned_ds[FeatureSelection]


# Replacing the Categorical Values with Numbers
featSelc_dataset['Status'].replace(statusToNum, inplace=True)
featSelc_dataset['Location_Type'].replace(loctypeToNum, inplace=True)
featSelc_dataset['Occurrence_Time'].replace(timeToNum, inplace=True)
scaler = preprocessing.StandardScaler()

scaled_df = scaler.fit_transform(featSelc_dataset)
scaled_df = pandas.DataFrame(scaled_df, columns = featSelc_dataset.columns)
print(scaled_df['Location_Type'].describe())
print(scaled_df['Occurrence_Time'].describe())
print(scaled_df['Status'].describe())
print(scaled_df['Hood_ID'].describe())
print(scaled_df.dtypes)


dependant_variable = 'Status'
x = scaled_df[scaled_df.columns.difference([dependant_variable])]
y = scaled_df[dependant_variable]

y=y.astype(int)

from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(x,y, test_size = 0.2)

lr = LogisticRegression(solver='lbfgs')
lr.fit(x,y)

from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print('The score of the 10 fold run is: ', score)

testY_predict = lr.predict(testX)
testY_predict.dtype
print(f"The 'Odds' \n{testY_predict.sum()}" )

from sklearn import metrics
labels = y.unique()
print(labels)
print("Accuracy:", metrics.accuracy_score(testY, testY_predict))
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n", confusion_matrix(testY, testY_predict, labels))


# Step 14
import joblib 
joblib.dump(lr, 'model_lr2.pkl')
print("Model dumped!")


# Step 15
model_columns = list(x.columns)
print(model_columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")

# Part B End

"""===========================================================================
    PART C) Train, Test data spliting – use numpy, sci-kit learn.
==========================================================================="""

# Part C Start



# Part C End

"""===========================================================================
    PART D) Managing imbalanced classes if needed. 
            Check here for info: 
                https://elitedatascience.com/imbalanced-classes
==========================================================================="""

# Part D Start



# Part D End