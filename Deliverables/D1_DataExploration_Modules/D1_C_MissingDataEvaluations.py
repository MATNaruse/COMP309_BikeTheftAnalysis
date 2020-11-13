# -*- coding: utf-8 -*-
"""
COMP309 Bike Theft Analysis
301 041 132 Trent B Minia
### ### ### Simon Ducuara

1) Data exploration: a complete review and analysis of the dataset including:
    c) Missing data evaluations – use pandas, numpy and any other python 
        packages

Created on Wed Nov 11 13:05:05 2020
"""
from D1_DataExploration_Modules.D1_A_LoadDescribeData import BikeData
import pandas as pd

class MissingData:
    @staticmethod
    def get_missing_data():
        missing_dataset = BikeData.get_dataset()
        return pd.isnull(missing_dataset)