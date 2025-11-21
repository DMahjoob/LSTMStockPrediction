from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import io
import base64

app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # last timestep
        out = self.fc1(out)
        out = self.fc2(out)
        return out

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

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop (equivalent to Keras .fit)
    model.train()
    epochs = 10
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Predict future prices
    model.eval()
    last_sequence = data_scaled[-sequence_length:]
    predictions = []
    for _ in range(future_days):
        input_sequence = torch.tensor(last_sequence.reshape(1, sequence_length, 1), dtype=torch.float32)
        with torch.no_grad():
            predicted_price = model(input_sequence)
        predictions.append(predicted_price.item())
        last_sequence = np.append(last_sequence[1:], [[predicted_price.item()]], axis=0)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
