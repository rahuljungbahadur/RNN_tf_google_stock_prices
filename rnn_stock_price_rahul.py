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

##Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60,1258):
    X_train.append(training_scaled[i-60:i, 0])
    y_train.append(training_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)


##Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

##Building the RNN

##Importing Keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

###

regressor = Sequential()

##Adding the first LSTM layer and Dropout layer

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

##Adding the second LSTM layer

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

##Adding the third LSTM layer

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

##Adding the fourth LSTM layer

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

##Adding the output layer
regressor.add(Dense(units = 1))

##Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

##Fitting the RNN

regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)

##Making predictions and visualizing the results


##