# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:06:00 2020

@author: Matthew
"""
# Step 5

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',30) # set the maximum width
# Load the dataset in a dataframe object 
df = pd.read_csv('D:/School/Fall2020/COMP309_BikeTheftAnalysis/Dataset/Bicycle_Thefts.csv')
# Explore the data check the column values
print(df.columns.values)
print (df.head())
print (df.info())
categories = []
for col, col_type in df.dtypes.iteritems():
     if col_type == 'O':
          categories.append(col)
     else:
          df[col].fillna(0, inplace=True)
print(categories)
print(df.columns.values)
print(df.head())
df.describe()
df.info()
#check for null values
print(len(df) - df.count())

# Step 7
include = ["Location_Type", "Premise_Type","Hood_ID", "Status"]
df_ = df[include]

print(df_.columns.values)
print(df_.head())
df_.describe()
df_.dtypes
df_['Location_Type'].unique()
df_['Premise_Type'].unique()
df_['Hood_ID'].unique()
df_['Status'].unique()
# check the null values
print(df_.isnull().sum())
print(len(df_) - df_.count())

# Step 8
df_.loc[:,("Location_Type", "Premise_Type", "Hood_ID", "Status")].dropna(axis=0,how='any',inplace=True) 
df_.info() 


# Step 9
categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     
print(categoricals)


# Step 10
df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=False)
print(df_ohe.head())
print(df_ohe.columns.values)
print(len(df_ohe) - df_ohe.count())


# Step 11
from sklearn import preprocessing
# Get column names first
names = df_ohe.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_ohe)
scaled_df = pd.DataFrame(scaled_df, columns=names)
print(scaled_df.head())
print(scaled_df.describe())
print(scaled_df.dtypes)


# Step 12
from sklearn.linear_model import LogisticRegression
dependent_variable = 'Status_RECOVERED'
# Another way to split the features
x = scaled_df[scaled_df.columns.difference([dependent_variable])]
x.dtypes
y = scaled_df[dependent_variable]
#convert the class back into integer
y = y.astype(int)
# Split the data into train test
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)
trainX.shape
testX.shape
trainY.shape
testY.shape
#build the model
lr = LogisticRegression(solver='lbfgs')
lr.fit(x, y)
# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)


# Step 13
testY_predict = lr.predict(testX)
testY_predict.dtype

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
labels = y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))


# Step 14
import joblib 
joblib.dump(lr, 'model_lr2.pkl')
print("Model dumped!")


# Step 15
model_columns = list(x.columns)
# print(model_columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
