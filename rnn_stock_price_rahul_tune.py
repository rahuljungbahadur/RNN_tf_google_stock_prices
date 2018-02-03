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
from keras.models import load_model

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
##saving the model
regressor.save(filepath = "RNN_tf_google_stock_prices.h5")
regressor = load_model(filepath = "RNN_tf_google_stock_prices.h5")
##Making predictions and visualizing the results

##Getting the test set for prices in 2017

test_set = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = test_set.iloc[:,1:2]


##getting the predicted stock prices
##for this one would need to combine both the training and test datasets so that while predicting the 2017 'open' data price the model can access the past 60 days of data
##concatenating both the data sets
data_total = pd.concat([data_train['Open'], test_set['Open']], axis = 0)

##to predict the stock prices of 2017 we need the data from the last 60
inputs = data_total[len(data_total) - len(test_set) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scale.transform(inputs)

##Converting this array into 3D for entering into the LSTM model
x_test = []
for i in range(60,80):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)

##Reshaping test_set to a 3D for entering it in LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
##predicting the prices
predict_stock_price = regressor.predict(x_test)
predict_stock_price = scale.inverse_transform(predict_stock_price)

##Calculating RMSE error
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predict_stock_price)) ##17.00



##Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predict_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Open Stock Price')
plt.legend()
plt.show()
plt.savefig('Google_stock_price.png', dpi = 300,orientation = 'portrait')

