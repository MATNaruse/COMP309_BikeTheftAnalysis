# -*- coding: utf-8 -*-
"""
COMP309 Bike Theft Analysis
### ### ### : Full Name

1) Data exploration: a complete review and analysis of the dataset including:
    d) Graphs and visualizations – use pandas, matplotlib, seaborn, 
        numpy and any other python packages, you also can use power BI desktop.

Created on Wed Nov 11 13:06:20 2020
"""
from D1_A_LoadDescribeData import BikeData
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class graph:
    dataset = BikeData.get_dataset()
    
    print(dataset.columns)
    
    x = dataset ['X']
    y = dataset ['Y']
    plt.plot(x,y)
    plt.show()