# -*- coding: utf-8 -*-
"""
COMP309 Bike Theft Analysis
### ### ### : Full Name

1) Data exploration: a complete review and analysis of the dataset including:
    d) Graphs and visualizations – use pandas, matplotlib, seaborn, 
        numpy and any other python packages, you also can use power BI desktop.

Created on Wed Nov 11 13:06:20 2020
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Graph:
    def __init__(self, dataset_in: pd.DataFrame, x_axis: str, y_axis :str):
        dataset = dataset_in
        x = dataset [x_axis]
        y = dataset [y_axis]
        plt.plot(x,y)
        plt.show()
        