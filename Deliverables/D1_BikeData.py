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
                       "Bike_Speed", "Lat", "Long", "Hood_ID"]


    def __init__(self):
        self._dataset: pd.DataFrame = None
        self._metadataset: pd.DataFrame = None

    def get_raw_dataset(self) -> pd.DataFrame:
        """
        Deliverable 1 - A
        Get DataFrame with Raw Data from 'Bicycle_Thefts.csv'
        
        Returns
        -------
        dataset : pd.DataFrame
            Created from 'Bicycle_Thefts.csv', Unmodified

        """
        if self._dataset is None:
            print("EMPTY! - FILLING _dataset!")
            path_dataset = os.path.join(Path(__file__).parents[2],
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
            path_dataset_meta = os.path.join(Path(__file__).parents[2], 
                                             "Dataset\Bicycle_Thefts_Metadata.csv")
            self._metadataset = pd.read_csv(path_dataset_meta)
        
        return self._metadataset
    
    
    def get_graph_raw(self,x_axis: str, y_axis :str):
        """
        Deliverable 1 - D

        Parameters
        ----------
        x_axis : str
            DESCRIPTION.
        y_axis : str
            DESCRIPTION.

        Returns
        -------
        plot : TYPE
            DESCRIPTION.

        """
        x = self.get_raw_dataset()[x_axis]
        y = self.get_raw_dataset()[y_axis]
        plot = plt.plot(x, y)
        plt.show()
        return plot
    
    

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
        
    