# -*- coding: utf-8 -*-
"""
COMP309 Bike Theft Analysis
### ### ### : Full Name
### ### ### : Full Name
### ### ### : Full Name
### ### ### : Full Name
300 549 638 : Matthew Naruse

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

import os, pandas
from pathlib import Path

# Modify this from True/False to show full dataset.describe() in console
pandas.set_option('display.expand_frame_repr', False)

"""===========================================================================
    PART A) Load and describe data elements (columns), provide descriptions 
            & types, ranges and values of elements as aproppriate. 
            – use pandas, numpy and any other python packages.
==========================================================================="""

# Part A Start

# Find Relative Paths 
# (from ../COMP309_BikeTheftAnalysis/Deliverables)
path_dataset = os.path.join(Path(__file__).parents[1], 
                                "Dataset\Bicycle_Thefts.csv")
path_dataset_meta = os.path.join(Path(__file__).parents[1], 
                                "Dataset\Bicycle_Thefts_Metadata.csv")

# Load Data
dataset = pandas.read_csv(path_dataset)
dataset_meta = pandas.read_csv(path_dataset_meta)

# Descriptions & Types
dataset.dtypes
print(dataset_meta)

# Ranges & Values
dataset.describe()

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


# Part C End

""" ==========================================================================
    PART D) Graphs and visualizations – use pandas, matplotlib, seaborn, 
            numpy and any other python packages, you also can use power
            BI desktop.
==========================================================================="""

# Part D Start


# Part D End







