# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:33:43 2018

@author: rahul
"""
###Recurrent Neural networks



##Data Preprocessing

##Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##Importing the training set
data_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = data_train.iloc[:, 1:2].values

##Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range = (0,1))

training_scaled = scale.fit_transform(training_set)

##Building the RNN


##Making predictions and visualizing the results


##