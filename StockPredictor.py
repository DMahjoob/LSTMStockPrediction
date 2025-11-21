from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import io
import base64

app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('student.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    ticker = request.form['ticker']
    period = request.form['period']
    future_days = {'1mo': 30, '6mo': 180, '1y': 365}.get(period, 30)  # Default to 30 days if invalid

    # Fetch stock data from Yahoo Finance
    df = yf.Ticker(ticker).history(period='10y')

    # Use the 'Close' column and drop missing values
    data = df[['Close']].dropna()

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Create sequences (30 days of history to predict the next day)
    sequence_length = 30
    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=10, verbose=0)

    # Predict future prices
    last_sequence = data_scaled[-sequence_length:]
    predictions = []
    for _ in range(future_days):
        input_sequence = last_sequence.reshape((1, sequence_length, 1))
        predicted_price = model.predict(input_sequence, verbose=0)
        predictions.append(predicted_price[0, 0])
        last_sequence = np.append(last_sequence[1:], predicted_price, axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Final predicted price after the chosen period
    final_price = predictions[-1][0]  # Get the last predicted price

    # Prepare predicted dates
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date, periods=future_days + 1, freq='D')[1:]

    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Historical Prices')
    plt.plot(future_dates, predictions, label=f'Predicted Prices ({period})', linestyle='--', color='orange')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)

    # Save the plot to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # Pass final predicted price to the template
    return render_template('result.html', ticker=ticker, period=period, plot_url=plot_url, final_price=final_price)
