# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

class BikeData:
    """
    Class that returns Deliverable 1 Related Data
    """
    
    FeatureColumns: [] = ["Primary_Offence", "Occurance_Time", "Location_Type",
                          "Bike_Type", "Status", "Hood_ID"]
    
    DropColumns: [] = ["X", "Y", "FID", "Occurrence_Date", "City", 
                       "Bike_Speed", "Lat", "Long"]


    def __init__(self):
        self._dataset: pd.DataFrame = None
        self._metadataset: pd.DataFrame = None

    def get_raw_dataset(self, recreate:bool = False) -> pd.DataFrame:
        """
        Deliverable 1 - A
        Get DataFrame with Raw Data from 'Bicycle_Thefts.csv'
        
        Returns
        -------
        dataset : pd.DataFrame
            Created from 'Bicycle_Thefts.csv', Unmodified

        """
        if self._dataset is None or recreate:
            print("EMPTY! - FILLING _dataset!")
            path_dataset = os.path.join(Path(__file__).parents[1],
                                        "Dataset\Bicycle_Thefts.csv")
            self._dataset = pd.read_csv(path_dataset)
            
        return self._dataset
        

    def get_raw_metadataset(self) -> pd.DataFrame:
        """
        Deliverable 1 - A
        Get DataFrame with Raw Data from 'Bicycle_Thefts_Metadata.csv'

        Returns
        -------
        dataset_meta : pd.DataFrame
            Created from 'Bicycle_Thefts_Metadata.csv', Unmodified
        """
        if self._metadataset is None:
            print("EMPTY! - FILLING _metadataset!")
            path_dataset_meta = os.path.join(Path(__file__).parents[1], 
                                             "Dataset\Bicycle_Thefts_Metadata.csv")
            self._metadataset = pd.read_csv(path_dataset_meta)
        
        return self._metadataset
       
    
    def get_clean_dataset(self) -> pd.DataFrame:
        self.get_raw_dataset()
        # ["Bike_Model", "Bike_Colour", "Cost_of_Bike"]
        self._dataset["Bike_Model"].fillna(value="??", inplace=True)
        self._dataset["Bike_Colour"].fillna(value="Unknown", inplace=True)
        self._dataset["Cost_of_Bike"].fillna(self._dataset["Cost_of_Bike"].mean(), inplace=True)
        print("CLEANED _dataset")
        return self._dataset


    def get_trimmed_dataset(self, dropColumns:[] = None) -> pd.DataFrame:
        """
                Get DataFrame with all useful* columns ONLY
        *Removed ["X","Y","FID", "Occurrence_Date", "City", "Bike_Speed", "Lat",
                 "Long","Hood_ID"]

        Parameters
        ----------
        dropColumns : [], optional, List of Column Names
            DESCRIPTION. The default is None.

        Returns
        -------
        dataset : pd.DataFrame
            DataFrame with Trimmed Columns from Raw Data
        """
        if not dropColumns:
            return self.get_raw_dataset().drop(columns=self.DropColumns)
        else:
            return self.get_raw_dataset().drop(columns=dropColumns)
        
    