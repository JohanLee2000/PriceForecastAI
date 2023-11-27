# layout
# First check to see if we have the stock in the database
# If we do, then we can just pull the data from the database
# If we don't, then we need to fetch the data from Yahoo Finance and then store it in the database
# using the stock ticker as the key 
# the database will store the last accessed date, the model, and the scaler
# Stored in a sqlite database
# Once we have the data, we can either update the model or train a new model and add it to the database

import sqlite3
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime, timedelta
import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM

# Database setup
db_connection = sqlite3.connect('stock_data.db')
cursor = db_connection.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_data (
        ticker TEXT PRIMARY KEY,
        last_accessed DATE,
        model BLOB,
        scaler BLOB
    )
''')
db_connection.commit()

# Function to fetch data from Yahoo Finance
def fetch_data(ticker):
    data = yf.download(ticker, start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'))
    return data

# Function to save model and scaler to database
def save_model(ticker, model, scaler):
    pickled_model = pickle.dumps(model)
    pickled_scaler = pickle.dumps(scaler)
    cursor.execute('''
        INSERT INTO stock_data (ticker, last_accessed, model, scaler) 
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET 
            last_accessed=excluded.last_accessed,
            model=excluded.model,
            scaler=excluded.scaler
    ''', (ticker, datetime.now(), pickled_model, pickled_scaler))
    db_connection.commit()

def train_lstm_model(data):
    # Assuming 'Adj Close' is the target variable
    stock_prices = data['Adj Close'].values.reshape(-1, 1)

    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    stock_prices_scaled = sc.fit_transform(stock_prices)

    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(60, len(stock_prices_scaled)):
        X_train.append(stock_prices_scaled[i-60:i, 0])
        y_train.append(stock_prices_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the data for LSTM (batch_size, timesteps, input_dim)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model to the training set
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    return model, sc

def predict_tomorrows_closing_lstm(ticker, model, sc):
    # Fetch the latest data
    latest_data = fetch_latest_data(ticker)

    # Use 'Adj Close' as the feature
    stock_prices = latest_data['Adj Close'].values.reshape(-1, 1)

    # Feature Scaling
    stock_prices_scaled = sc.transform(stock_prices)

    # Create input data structure for LSTM
    inputs = []
    for i in range(len(stock_prices_scaled) - 60, len(stock_prices_scaled)):
        inputs.append(stock_prices_scaled[i-60:i, 0])
    inputs = np.array(inputs)

    # Reshape data for LSTM (batch_size, timesteps, input_dim)
    inputs = np.reshape(inputs, (1, inputs.shape[0], 1))

    # Make predictions
    predicted_scaled_price = model.predict(inputs)

    # Inverse transform to get the original scale
    predicted_price = sc.inverse_transform(predicted_scaled_price.reshape(-1, 1))

    return predicted_price[0, 0]


def train_model(data):
    # Step 1: Data Cleaning
    # In financial datasets, it's common to have no missing data for these features.
    # However, if there are missing values, handle them appropriately.
    data.fillna(method='ffill', inplace=True)  # Forward fill for time series data

    # Step 2: Feature Engineering
    # Create new financial indicators/features that could be beneficial for the model
    # Example: Moving averages, daily return, etc.
    data['Daily Return'] = data['Adj Close'].pct_change()  # Daily return
    data.dropna(inplace=True)  # Drop NA values created by pct_change

    # Step 3: Data Transformation
    # Normalize/Standardize features
    scaler = StandardScaler()
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Daily Return']
    data[features] = scaler.fit_transform(data[features])

    # Step 4: Splitting the Data
    # Use 'Adj Close' or another feature as the target variable
    X = data[features]  # All features except the target
    y = data['Adj Close']  # Predicting adjusted closing price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Train the Model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Optionally: Evaluate the model using X_test and y_test



    return model, scaler

# Function to fetch the latest data
def fetch_latest_data(ticker):
    # Fetch the most recent data (e.g., the last 60 days)
    recent_data = yf.download(ticker, period="60d", interval="1d")
    return recent_data

# Function to load model and scaler from the database
def load_model_and_scaler_and_time(ticker):
    cursor.execute("SELECT model, scaler, last_accessed FROM stock_data WHERE ticker=?", (ticker,))
    result = cursor.fetchone()
    if result:
        model, scaler, date = pickle.loads(result[0]), pickle.loads(result[1]), result[2]
        return model, scaler, date
    else:
        raise Exception("Model and scaler not found for ticker:", ticker)

# Function to predict tomorrow's closing
def predict_tomorrows_closing(ticker):
    # Load model and scaler
    model, scaler, last_accessed = load_model_and_scaler_and_time(ticker)
    print(last_accessed)
    # cast scaler to scaler object
    datetime_object = datetime.strptime(last_accessed, '%Y-%m-%d %H:%M:%S.%f')
    print(datetime_object)
    # if the data is more than a day old, then we need to update the model
    print(datetime.now() - datetime_object)
    if (datetime.now() - datetime_object) > timedelta(days=1):
        # If the data is more than a day old, update the model
        # Fetch the latest data
        latest_data = fetch_latest_data(ticker)
        
        model, scaler = train_model(latest_data)

        latest_features = scaler.transform(latest_data.iloc[-1:])
        
        # Predict tomorrow's closing
        predicted_closing = model.predict(latest_features)
        return predicted_closing[0]

    # Fetch and preprocess latest data
    # Apply the same preprocessing steps as during training
    # (e.g., feature engineering, scaling using the loaded scaler)
    # ...

    # Prepare the input for prediction (typically the latest available data point)
    # Ensure the input format is correct (e.g., 2D array for scikit-learn models)
    if scaler is not None:
        latest_data = fetch_latest_data(ticker)
        latest_data['Daily Return'] = latest_data['Adj Close'].pct_change()  # Daily return
        print(latest_data.iloc[-1:])
        latest_features = scaler.transform(latest_data.iloc[-1:])
        predicted_closing = model.predict(latest_features)
        return predicted_closing[0]
    else:
        raise Exception("Scaler is not available for ticker:", ticker)
    


# Main function to handle stock data
def handle_stock_data(ticker):
    cursor.execute("SELECT * FROM stock_data WHERE ticker=?", (ticker,))
    result = cursor.fetchone()

    if result:
        # If data exists, use it
        print(f"Data for {ticker} found in database.")
        # Additional logic can be added here to update model if needed
    else:
        # If data doesn't exist, fetch and store it
        print(f"Fetching data for {ticker} from Yahoo Finance.")
        data = fetch_data(ticker)
        print(data)
        model, scaler = train_model(data)
        save_model(ticker, model, scaler)  

# Example usage
ticker = 'AAPL'
handle_stock_data(ticker)

# Train and save LSTM model
data = fetch_data(ticker)
lstm_model, lstm_scaler = train_lstm_model(data)
save_model(ticker, lstm_model, lstm_scaler)

# Predict with LSTM model
predicted_closing_lstm = predict_tomorrows_closing_lstm(ticker, lstm_model, lstm_scaler)
print(f"Predicted closing price for {ticker} tomorrow using LSTM: {predicted_closing_lstm}")