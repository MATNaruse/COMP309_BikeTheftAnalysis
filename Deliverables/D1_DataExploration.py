# -*- coding: utf-8 -*-
"""
COMP309 Bike Theft Analysis
301 063 251 : Arthur Batista
300 549 638 : Matthew Naruse
301 041 132 : Trent B Minia
300 982 276 : Simon Ducuara
### ### ### : Full Name

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
from D1_BikeData import BikeData

# Modify this from True/False to show full dataset.describe() in console
pandas.set_option('display.expand_frame_repr', True)

""" ==========================================================================
    INITIALIZING BikeData CLASS
==========================================================================="""

BikeData = BikeData()

"""===========================================================================
    PART A) Load and describe data elements (columns), provide descriptions 
            & types, ranges and values of elements as aproppriate. 
            – use pandas, numpy and any other python packages.
==========================================================================="""

# Part A Start

dataset = BikeData.get_raw_dataset()
metadataset = BikeData.get_raw_metadataset()

# Part A End

""" ==========================================================================
    PART B) Statistical assessments including means, averages, correlations
==========================================================================="""

# Part B Start

desc_ds = dataset.describe()

# Part B End

""" ==========================================================================
    PART C)  Missing data evaluations – use pandas, numpy and any other 
            python packages
==========================================================================="""

# Part C Start

miss_ds = pandas.isnull(dataset).sum()

# X                      0
# Y                      0
# FID                    0
# Index_                 0
# event_unique_id        0
# Primary_Offence        0
# Occurrence_Date        0
# Occurrence_Year        0
# Occurrence_Month       0
# Occurrence_Day         0
# Occurrence_Time        0
# Division               0
# City                   0
# Location_Type          0
# Premise_Type           0
# Bike_Make              0
# Bike_Model          8141
# Bike_Type              0
# Bike_Speed             0
# Bike_Colour         1729
# Cost_of_Bike        1536
# Status                 0
# Hood_ID                0
# Neighbourhood          0
# Lat                    0
# Long                   0

# Only need to clean ["Bike_Model", "Bike_Colour", "Cost_of_Bike"]

# Part C End

""" ==========================================================================
    PART D) Graphs and visualizations – use pandas, matplotlib, seaborn, 
            numpy and any other python packages, you also can use power
            BI desktop.
==========================================================================="""

# Part D Start

# BikeData.get_graph_raw("Neighbourhood", "Status")

# Part D End

