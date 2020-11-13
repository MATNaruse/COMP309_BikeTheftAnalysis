# -*- coding: utf-8 -*-
"""
COMP309 Bike Theft Analysis
### ### ### : Full Name
### ### ### : Full Name
### ### ### : Full Name
### ### ### : Full Name
300 549 638 : Matthew Naruse
301 041 132 : Trent B Minia
300 982 276 : Simon Ducuara

1) Data exploration: a complete review and analysis of the dataset including:
    a) Load and describe data elements (columns), provide descriptions 
        & types, ranges and values of elements as aproppriate. 
        – use pandas, numpy and any other python packages.

    b) Statistical assessments including means, averages, correlations

    c) Missing data evaluations – use pandas, numpy and any other python 
        packages

    d) Graphs and visualizations – use pandas, matplotlib, seaborn, numpy 
        and any other python packages, you also can use power BI desktop.

Created on Wed Nov 11 13:08:52 2020
"""

# External Imports
import os, pandas
from pathlib import Path

# Local Imports
from D1_DataExploration_Modules.D1_A_LoadDescribeData import BikeData
from D1_DataExploration_Modules.D1_C_MissingDataEvaluations import MissingData

# Modify this from True/False to show full dataset.describe() in console
pandas.set_option('display.expand_frame_repr', False)

"""===========================================================================
    PART A) Load and describe data elements (columns), provide descriptions 
            & types, ranges and values of elements as aproppriate. 
            – use pandas, numpy and any other python packages.
==========================================================================="""

# Part A Start

dataset = BikeData.get_dataset()
dataset_meta = BikeData.get_metadataset()

# Part A End

""" ==========================================================================
    PART B) Statistical assessments including means, averages, correlations
==========================================================================="""

# Part B Start


# Part B End

""" ==========================================================================
    PART C)  Missing data evaluations – use pandas, numpy and any other 
            python packages
==========================================================================="""

# Part C Start

missing_dataset = MissingData.check_missing_data()

# Part C End

""" ==========================================================================
    PART D) Graphs and visualizations – use pandas, matplotlib, seaborn, 
            numpy and any other python packages, you also can use power
            BI desktop.
==========================================================================="""

# Part D Start


# Part D End







