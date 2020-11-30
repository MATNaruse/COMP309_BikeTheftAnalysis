# -*- coding: utf-8 -*-
"""
COMP309 Bike Theft Analysis
301 041 132 : Trent B Minia
300 982 276 : Simon Ducuara

1) Data Modelling:
    a) Data transformations – includes handling missing data, categorical 
        data management, data normalization and standardizations as needed.

"""

from Deliverables.D1_DataExploration_Modules.D1_A_LoadDescribeData import BikeData

dataset = BikeData.get_dataset()
dataset_meta = BikeData.get_metadataset()

# fillna = Finds nan/NA and replaces with given value
dataset["Bike_Colour"].fillna(value="N/A", inplace=True)

# Cost of Bike, replace nan with Average
# *Round Later
dataset["Cost_of_Bike"].fillna(dataset["Cost_of_Bike"].mean(), inplace=True)

mod_dataset = BikeData.get_dataset().drop(columns=["X","Y","FID", "Occurrence_Date", "City", "Bike_Speed", "Lat", "Long", "Hood_ID"])


# KP?? 0	Index
# KP?? 1	event_unique_id
# KEEP 2	*Primary_Offence
# TOSS 3	Occurrence_Date
# KEEP 4	Occurrence_Year
# KEEP 5	Occurrence_Month
# KEEP 6	Occurrence_Day
# KEEP 7	*Occurrence_Time
# KEEP 8	Division
# TOSS 9	City
# KEEP 10	*Location_Type
# KEEP 11	Premise_Type
# KEEP 12	Bike_Make
# KEEP 13	Bike_Model -> DummyData
# KEEP 14	*Bike_Type
# TOSS 15	Bike_Speed
# KEEP 16	Bike_Colour
# KEEP 17	Cost_of_Bike
# KEEP 18	*Status
# TOSS 19	Lat
# TOSS 20	Long
# KEEP ##   *Neighbourhood
# TOSS ##   Hood_ID