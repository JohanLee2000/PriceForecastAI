# Using LTSM (Long-Term Short Memory Network)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

company = 'AAPL'
start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

data = yf.download(company, start=start, end=end)
# data = web.DataReader(company, 'yahoo', start, end)

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

#Test accuracy
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

# test_data = web.DataReader(company, 'yahoo', test_start, test_end)
test_data = yf.download(company, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'],test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#Make predictions
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()


#Predict next day
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# # Using LTSM (Long-Term Short Memory Network)
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# # %matplotlib inline

# dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
# dataset_train.head()

# training_set = dataset_train.iloc[:,1:2].values

# # print(training_set)
# # print(training_set.shape)

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range = (0, 1))
# scaled_training_set = scaler.fit_transform(training_set)


# X_train = []
# Y_train = []
# for i in range(60, 1258):
#     X_train.append(scaled_training_set[i-60:i, 0])
#     Y_train.append(scaled_training_set[i, 0])
# X_train = np.array(X_train)
# Y_train = np.array(Y_train)

# X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))
# #7
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import Dropout

# regressor = Sequential()

# regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# regressor.add(Dropout(0.2))

# regressor.add(LSTM(units = 50, return_sequences = True))
# regressor.add(Dropout(0.2))


# regressor.add(LSTM(units = 50, return_sequences = True))
# regressor.add(Dropout(0.2))


# regressor.add(LSTM(units = 50))
# regressor.add(Dropout(0.2))

# regressor.add(Dense(units =  1))
# #
# regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)

# dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
# actual_stock_price = dataset_test.iloc[:,1:2].values

# dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

# inputs = inputs.reshape(-1, 1)
# inputs = scaler.transform(inputs)

# X_test = []
# for i in range(60, 80):
#     X_test.append(inputs[i-60:i, 0])
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# predicted_stock_price = regressor.predict(X_test)
# predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# #
# plt.plot(actual_stock_price, color = 'red', label = 'Actual Google Stock Price')
# plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
# plt.title('Google Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Google Stock Price')
# plt.legend()
# plt.show()