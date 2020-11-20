# -*- coding: utf-8 -*-
"""
COMP309 Bike Theft Analysis
301 063 251 : Arthur Batista
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

# External Imports
import os, pandas
from pathlib import Path

# Local Imports
from D1_DataExploration_Modules.D1_A_LoadDescribeData import BikeData
from D1_DataExploration_Modules.D1_B_StatisticalAssessments import Statistical
from D1_DataExploration_Modules.D1_D_GraphsVisualizations import Graph

# Modify this from True/False to show full dataset.describe() in console
pandas.set_option('display.expand_frame_repr', True)

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

statistic_dataset = Statistical.get_statistical_dataset()

#Found out it is not necessary display statistical assessments for metadataset
#statistic_metadataset = Statistical.get_statistical_metadataset()

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

FirstGraph = Graph(dataset, "X", "Y")

# Part D End

