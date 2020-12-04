# -*- coding: utf-8 -*-
# External Imports
import traceback, joblib, os
from pathlib import Path
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score

"""
Support Functions
"""

def get_cat_col(df: pd.DataFrame, df_label:str = None, fillna:bool = False) -> []:
    """
    Returns list of columns that contain Categorical Values

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check
    df_label : str, optional
        Label for console output. The default is None.
    fillna : bool, optional
        Boolean to fill numerical values while iterating. The default is False.

    Returns
    -------
    []
        List of column names that contain Categorical Values.

    """
    categorical_columns = []
    for col, col_type in df.dtypes.iteritems():
        if col_type == 'O':
            categorical_columns.append(col)
        elif fillna and col_type != 'O':
            df[col].fillna(0, inplace=True)
            
    out_msg = "\nCategorical columns found"
    if df_label:
       out_msg += " for " + df_label
    print(out_msg)
    print("=" * len(out_msg))
    
    for col in categorical_columns:
        print(f"\t- {col}")
    return categorical_columns

def disp_col_w_missing(df: pd.DataFrame, df_label:str = None, colList:[] = None, threshold:int = 0):
    """
    Displays all columns with missing values

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check
    df_label : str, optional
        Label for console output. The default is None.
    colList : [], optional
        List of specific column names to check. The default is None.
    threshold : int, optional
        Maximum number of missing values to ignore. The default is 0.

    Returns
    -------
    None.

    """
    col_w_missing = None
    if colList:
        col_w_missing = pd.DataFrame(df[colList].isnull().sum())
    else:
        col_w_missing = pd.DataFrame(df.isnull().sum())
    
    out_msg = "\nSum of Missing Values"
    if df_label:
        out_msg += " for " + df_label
    
    print(out_msg)
    print("=" * len(out_msg))
    print(col_w_missing[col_w_missing.iloc[:,0] > 0])
    
def gen_json_testset(df: pd.DataFrame, rando_num:int = 0, no_status:bool = False) -> str:
    """
    Generate a json string of existing data for Postman

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains data to test from.
    rando_num : int, optional
        Number of Random Rows to pull. The default is 0, which will use entire DataFrame.
    no_status : bool, optional
        Boolean whether to include the 'Status' column. The default is False.

    Returns
    -------
    str
        Json-formatted string to paste into Postman.

    """
    local_df = df
    if no_status:
        local_df = df.drop("Status", axis=1)
    if rando_num:
        rando_df = local_df.sample(n=rando_num)
        return rando_df.to_json(orient="records")
    return local_df.to_json(orient="records")

def gen_json_dummy(df: pd.DataFrame, pinColValue: [], size:int) -> str:
    """
    Generate a json string with specific column values for Postman
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains data to test from.
    pinColValue : []
        List of (ColumnName, ColumnValue) tuples 
    size : int
        Number of Samples to return

    Returns
    -------
    str
        Json-formatted string to paste into Postman.

    """
    local_df = df.sample(size)
    colValDict: {} = {}
    for tup in pinColValue:
        colValDict[tup[0]] = tup[1]
    
    for key, val in colValDict.items():
        print(f"{key} | {val}")
        local_df[key].replace(regex="^.*$", value=val, inplace=True)
    
    local_df = local_df.drop('Status', axis=1)
    return local_df.to_json(orient="records")
 

"""
Data Exploration
"""

# Loading Data
bikedata = pd.read_csv(os.path.join(Path(__file__).parents[1],
                                        "Dataset\Bicycle_Thefts.csv"))
# Quick View of Data
print(bikedata.columns.values)
print(bikedata.info())
print(bikedata.describe())

print("\nActual Recovery Numbers")
print("=======================")
print(bikedata['Status'].value_counts())


# Trimming irrelevant columns
bikedata = bikedata.drop(["X", "Y", "FID", "Index_", "event_unique_id"], axis=1)

# Sorting Categorical/String Varible Columns
categorical_columns = get_cat_col(bikedata,"bikedata", True)

# Finding out which columns have missing values 
# *Note: Already 'cleaned' numerical values in get_cat_col
disp_col_w_missing(bikedata, "bikedata", categorical_columns)

# As we are CURRENTLY not focusing on Bike_Model and Bike_Colour, don't need
#   to worry about them for now (?)

"""
Data Modeling
"""
# List of Features to focus on
FeatureSelection = ["Location_Type", "Premise_Type", "Division", "Status", "Neighbourhood"]
FS_bikedata = bikedata[FeatureSelection]

# Transforming 'Status' into binary [Recovered OR Stolen/Unknown]
FS_bikedata['Status'] = [1 if s=='RECOVERED' else 0 for s in FS_bikedata['Status']]
    
# Getting Categorical Columns for Dummy Generation
FS_bikedata_cat_col = get_cat_col(FS_bikedata, "FS_bikedata")
FS_bikedata_dumm = pd.get_dummies(FS_bikedata, columns=FS_bikedata_cat_col, dummy_na=False)

print("\nConfirming Missing Data(?):\n===========================")
print(len(FS_bikedata_dumm) - FS_bikedata_dumm.count())


# Managing Imbalanced Classes
bd_major = FS_bikedata_dumm[FS_bikedata_dumm.Status==0]
bd_minor = FS_bikedata_dumm[FS_bikedata_dumm.Status==1]

bd_major_downsampled= resample(bd_major, replace=False, n_samples=len(bd_minor), random_state=123)

bd_downsampled = pd.concat([bd_major_downsampled, bd_minor])

print(bd_downsampled.Status.value_counts())


# Splitting Data for Train/Test
y = bd_downsampled.Status
x = bd_downsampled.drop('Status', axis=1)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size =0.2)

"""
Predictive Model Building
"""

# Building the Model
lr = LogisticRegression(solver="lbfgs", max_iter=10000)
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
print("\nAccuracy: %d%%" % ((accuracy_score(yTest, yTest_predict)) * 100))

# Confusion Matrix
labels = y.unique() # [0, 9]
print("\nConfusion Matrix")
print('================')
print(confusion_matrix(yTest, yTest_predict, labels=labels))


"""
Model Dumping
"""

joblib.dump(lr, "model_lr_new.pkl")
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')

print("Model and Model Columns Dumped!")

print("\nModel Columns:")
for col in model_columns:
    print(f"\t- {col}")


   
"""
Deploying the Model
"""

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json

            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))
            count_0 = prediction.count(0)
            count_1 = prediction.count(1)
            
            out_json = {'predictions': str(prediction), 
                            'NOT Recovered': count_0, 
                            'Recovered': count_1}
            print(out_json)
            return jsonify(out_json)

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Model Required')
        return ('Model Not Found')

if __name__ == '__main__':
    
    print("Loading Model...")
    lr = joblib.load(os.path.join(Path(__file__).parents[0],'model_lr.pkl'))
    print('...Model loaded!')
    
    print("Loading Model Columns...")
    model_columns = joblib.load(os.path.join(Path(__file__).parents[0],'model_columns.pkl'))
    print('....Model columns loaded!')
    
    
    app.run(port=12345, debug=True)