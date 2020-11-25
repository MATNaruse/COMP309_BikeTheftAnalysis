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


class BikeData:
    
    
    FeatureColumns: [] = ["Primary_Offence", "Occurance_Time", "Location_Type",
                          "Bike_Type", "Status", "Neighbourhood"]
    
    @staticmethod
    def get_dataset() -> pandas.DataFrame:
        """
        DEPRECATED - Use get_raw_dataset()

        Returns:
        pandas.DataFrame : Created from 'Bicycle_Thefts.csv'

        """
        return BikeData.get_raw_dataset()
    
    @staticmethod
    def get_metadataset() -> pandas.DataFrame:
        """
        DEPRECATED - Use get_raw_metadataset()

        Returns:
        pandas.DataFrame : Created from 'Bicycle_Thefts_Metadata.csv'

        """
        return BikeData.get_raw_metadataset()

    @staticmethod
    def get_raw_dataset() -> pandas.DataFrame:
        """
        Get DataFrame with Raw Data from 'Bicycle_Thefts.csv'

        Returns
        -------
        dataset : pandas.DataFrame
            Created from 'Bicycle_Thefts.csv', Unmodified

        """
        path_dataset = os.path.join(Path(__file__).parents[2],
                                    "Dataset\Bicycle_Thefts.csv")
        dataset = pandas.read_csv(path_dataset)
        return dataset
        
    @staticmethod
    def get_raw_metadataset() -> pandas.DataFrame:
        """
        Get DataFrame with Raw Data from 'Bicycle_Thefts_Metadata.csv'

        Returns
        -------
        dataset_meta : pandas.DataFrame
            Created from 'Bicycle_Thefts_Metadata.csv', Unmodified
        """
        path_dataset_meta = os.path.join(Path(__file__).parents[2], 
                                         "Dataset\Bicycle_Thefts_Metadata.csv")
        dataset_meta = pandas.read_csv(path_dataset_meta)
        return dataset_meta
    
    @staticmethod
    def get_trimmed_dataset() -> pandas.DataFrame:
        """
        Get DataFrame with all useful* columns ONLY
        *Removed ["X","Y","FID", "Occurrence_Date", "City", "Bike_Speed", "Lat",
                 "Long","Hood_ID"]

        Returns
        -------
        dataset : pandas.DataFrame
            DataFrame with Trimmed Columns from Raw Data

        """
        return BikeData.get_raw_dataset().drop(columns=["X","Y","FID", 
                                                        "Occurrence_Date", 
                                                        "City", "Bike_Speed", 
                                                        "Lat", "Long", 
                                                        "Hood_ID"])
    