# -*- coding: utf-8 -*-
"""
COMP309 Bike Theft Analysis
300 549 638 : Matthew Naruse

1) Data exploration: a complete review and analysis of the dataset including:
    a) Load and describe data elements (columns), provide descriptions & types, 
        ranges and values of elements as aproppriate. 
        – use pandas, numpy and any other python packages.

Created on Wed Nov 11 12:25:33 2020
"""

import os, pandas
from pathlib import Path

# Find Relative Paths
path_dataset = os.path.join(Path(__file__).parents[2], "Dataset\Bicycle_Thefts.csv")
path_dataset_meta = os.path.join(Path(__file__).parents[2], "Dataset\Bicycle_Thefts_Metadata.csv")

# Load Data
dataset = pandas.read_csv(path_dataset)
dataset_meta = pandas.read_csv(path_dataset_meta)

@staticmethod
def test():
    print("It Works!")