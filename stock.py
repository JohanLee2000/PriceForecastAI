# Using LTSM (Long-Term Short Memory Network)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

import tkinter as tk

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def run_and_build_model(stock_entry, end_day_entry, days_to_compare_entry, prediction_label):

    #Gather input fields
    company = stock_entry.get()
    end_day = end_day_entry.get()
    days_to_compare = days_to_compare_entry.get()

    start = dt.datetime.strptime(end_day, "%Y,%m,%d")
    end = dt.datetime.today()

    #Download data
    data = yf.download(company, start=start, end=end)
    # print(data)

    #Scale data
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

    prediction_days = int(days_to_compare)
    #Prepare data for model to train on
    x_train = []
    y_train = []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #Create model and feed data, train it
    model = Sequential()

    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    #Test accuracy
    test_start = start
    test_end = dt.datetime.now()

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

    #Plot data on graph
    plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
    plt.plot(predicted_prices[20:], color="green", label=f"Predicted {company} Price")
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
    prediction_label.config(text=f"Prediction for tommorow: {prediction[0][0]}")

def main():
    #Create GUI
    window = tk.Tk()
    window.title("PriceForcast AI")

    stock_label = tk.Label(text="Enter the stock ticker: ")
    stock_entry = tk.Entry()
    stock_label.pack()
    stock_entry.pack()

    end_day_label = tk.Label(text="Enter the end day (YYYY,MM,DD): ")
    end_day_entry = tk.Entry()
    end_day_label.pack()
    end_day_entry.pack()

    days_to_compare_label = tk.Label(text="Enter the number of days to compare the prediction against: ")
    days_to_compare_entry = tk.Entry()
    days_to_compare_label.pack()
    days_to_compare_entry.pack()

    # Label for displaying prediction
    prediction_label = tk.Label(window, text="")
    prediction_label.pack()
    
    #Buttons
    button = tk.Button(window, text = "Predict", command=lambda: run_and_build_model(stock_entry, end_day_entry, days_to_compare_entry, prediction_label))
    button.pack()

    window.mainloop()

main()