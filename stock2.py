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
predicted_closing = predict_tomorrows_closing(ticker)
print(f"Predicted closing price for {ticker} tomorrow: {predicted_closing}")