# -*- coding: utf-8 -*-
"""
COMP309 Bike Theft Analysis
301 063 251 : Arthur Batista

1) Data exploration: a complete review and analysis of the dataset including:
    b) Statistical assessments including means, averages, correlations

Created on Wed Nov 11 13:03:35 2020
"""
#Import BikeData from D1_A_LoadDescribeData
from D1_DataExploration_Modules.D1_A_LoadDescribeData import BikeData

#Defining Statistical class
class Statistical:
    
    #method to display statistical assessments of dataset
    @staticmethod
    def get_statistical_dataset():
        statistics_dataset = BikeData.get_dataset()
        
        return statistics_dataset.describe()
    
    #method to display statistical assessments of metadataset
    #Not really necessary for this example
    @staticmethod
    def get_statistical_metadataset():
        statistics_metadataset = BikeData.get_metadataset()
        
        return statistics_metadataset.describe()
        


