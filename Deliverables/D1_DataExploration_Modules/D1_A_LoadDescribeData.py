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

pandas.set_option('display.expand_frame_repr', False)

# Find Relative Paths 
# (from ../COMP309_BikeTheftAnalysis/Deliverables/D1_DataExploration)
path_dataset = os.path.join(Path(__file__).parents[2], "Dataset\Bicycle_Thefts.csv")
path_dataset_meta = os.path.join(Path(__file__).parents[2], "Dataset\Bicycle_Thefts_Metadata.csv")

# Load Data
dataset = pandas.read_csv(path_dataset)
dataset_meta = pandas.read_csv(path_dataset_meta)

# Descriptions & Types
dataset.dtypes
#print(dataset_meta)

# Ranges & Values
dataset.describe()


class BikeData:
    
    @staticmethod
    def get_dataset():
        path_dataset = os.path.join(Path(__file__).parents[2], "Dataset\Bicycle_Thefts.csv")
        dataset = pandas.read_csv(path_dataset)
        return dataset
    
    @staticmethod
    def get_metadataset():
        path_dataset_meta = os.path.join(Path(__file__).parents[2], "Dataset\Bicycle_Thefts_Metadata.csv")
        dataset_meta = pandas.read_csv(path_dataset_meta)
        return dataset_meta
