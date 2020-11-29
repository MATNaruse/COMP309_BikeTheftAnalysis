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
import matplotlib.pyplot as plt

# Local Imports
from D1_DataExploration_Modules.D1_BikeData import BikeData


""" ==========================================================================
    INITIALIZING BikeData CLASS
==========================================================================="""

BikeData = BikeData()
raw_dataset = BikeData.get_raw_dataset()

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

FeatureSelection = ["Location_Type", "Status", "Hood_ID"]

# ds_norm = (trimmed_ds["Neighbourhood"] - trimmed_ds["Neighbourhood"].min()) / (trimmed_ds["Neighbourhood"].max() - trimmed_ds["Neighbourhood"].min())

def values_to_numbers(lValues:[], reverse:bool = False) -> {}:
    out_dict = {}
    for value in list(lValues):
        if(reverse):
            out_dict[value] = list(lValues).index(value)
        else:
            out_dict[list(lValues).index(value)] = value
    return out_dict


test = values_to_numbers(FeatureSelection)
nbhd_toNumbers = values_to_numbers(trimmed_ds['Neighbourhood'].unique())
locType_toNum = values_to_numbers(raw_dataset['Location_Type'].unique(), True)
status_toNum = values_to_numbers(raw_dataset['Status'].unique(), True)
d = pandas.DataFrame(
    {'Column_Name':trimmed_ds.columns.values, 
     'NuMbErS': np.random.randn(len(trimmed_ds.columns.values))})

featSelc_dataset = raw_dataset[FeatureSelection]

featSelc_dataset['Status'].replace(status_toNum, inplace=True)
featSelc_dataset['Location_Type'].replace(locType_toNum, inplace=True)
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