import pandas as pd

# Load the dataset
data = pd.read_csv("stocks.csv")
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Clean column names
data.columns = data.columns.str.strip()

# Convert columns to numeric (coerce errors to NaN if needed)
data['2 Week Prediction (USD)'] = pd.to_numeric(data['2 Week Prediction (USD)'], errors='coerce')
data['2 Week Actual Price (USD)'] = pd.to_numeric(data['2 Week Actual Price (USD)'], errors='coerce')
data['Current Price 11/20 (USD)'] = pd.to_numeric(data['Current Price 11/20 (USD)'], errors='coerce')

# Calculate changes
data['Predicted Change'] = data['2 Week Prediction (USD)'] - data['Current Price 11/20 (USD)']
data['Actual Change'] = data['2 Week Actual Price (USD)'] - data['Current Price 11/20 (USD)']

# Determine prediction direction
data['Predicted Up'] = data['Predicted Change'] > 0
data['Actual Up'] = data['Actual Change'] > 0
data['Correct Direction'] = data['Predicted Up'] == data['Actual Up']

# Calculate within 5% range
data['Within 5%'] = abs(data['2 Week Prediction (USD)'] - data['2 Week Actual Price (USD)']) / data['2 Week Actual Price (USD)'] <= 0.05

# Calculate statistical significance (e.g., >10% difference)
data['Significant Difference'] = abs(data['2 Week Prediction (USD)'] - data['2 Week Actual Price (USD)']) / data['2 Week Actual Price (USD)'] > 0.1
data['Error'] = data['2 Week Prediction (USD)'] - data['2 Week Actual Price (USD)']

# Metrics
predicted_up_correct = data[data['Predicted Up'] & data['Actual Up']].shape[0]
predicted_up_total = data[data['Predicted Up']].shape[0]
percent_up_correct = (predicted_up_correct / predicted_up_total) * 100 if predicted_up_total > 0 else 0

predicted_down_correct = data[~data['Predicted Up'] & ~data['Actual Up']].shape[0]
predicted_down_total = data[~data['Predicted Up']].shape[0]
percent_down_correct = (predicted_down_correct / predicted_down_total) * 100 if predicted_down_total > 0 else 0

within_5_percent = data['Within 5%'].sum()
percent_within_5 = (within_5_percent / data.shape[0]) * 100

significant_diff = data['Significant Difference'].sum()
percent_sig_dff = (significant_diff / data.shape[0]) * 100 if significant_diff > 0 else 0

# Print results
print(f"Percentage of stocks predicted to go up that actually went up: {percent_up_correct:.2f}%")
print(f"Percentage of stocks predicted to go down that actually went down: {percent_down_correct:.2f}%")
print(f"Percentage of stocks within 5% of their predicted value: {percent_within_5:.2f}%")
print(f"Percentage of stocks with statistically significant (>10%) differences: {percent_sig_dff:.2f}%")

# Filter stocks that were predicted to go up and actually went up
predicted_up_and_went_up = data[(data['Predicted Up']) & (data['Actual Up'])]

# Filter stocks that were predicted to go down and actually went down
predicted_down_and_went_down = data[(~data['Predicted Up']) & (~data['Actual Up'])]

# Display results
print("\n\nStocks that were predicted to go up and actually went up:")
print(predicted_up_and_went_up[['Ticker', 'Current Price 11/20 (USD)', '2 Week Prediction (USD)', '2 Week Actual Price (USD)']])

print("\nStocks that were predicted to go down and actually went down:")
print(predicted_down_and_went_down[['Ticker', 'Current Price 11/20 (USD)', '2 Week Prediction (USD)', '2 Week Actual Price (USD)']])

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# Order tickers for consistent graphing
ticker_order = data['Ticker']

# ---------- 1. Predicted vs Actual Prices ----------
plt.figure(figsize=(14,7))
sns.lineplot(x=ticker_order, y=data['2 Week Actual Price (USD)'], label='Actual Price')
sns.lineplot(x=ticker_order, y=data['2 Week Prediction (USD)'], label='Predicted Price')
plt.xticks(rotation=90)
plt.title("Predicted vs Actual Stock Prices (2-Week Horizon)", fontsize=16)
plt.xlabel("Stock")
plt.ylabel("Price (USD)")
plt.tight_layout()
plt.show()


# ---------- 2. Prediction Error Over Stocks ----------
plt.figure(figsize=(14,7))
sns.lineplot(x=ticker_order, y=data['Error'])
plt.axhline(0, linestyle='--', color='black')
plt.xticks(rotation=90)
plt.title("Prediction Error per Stock", fontsize=16)
plt.xlabel("Stock")
plt.ylabel("Prediction Error (USD)")
plt.tight_layout()
plt.show()


# ---------- 3. Residual Distribution Histogram ----------
plt.figure(figsize=(10,6))
sns.histplot(data['Error'], bins=20, kde=True)
plt.title("Residual Distribution (Predicted - Actual)", fontsize=16)
plt.xlabel("Error (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ---------- 4. Scatter Plot: Actual vs Predicted ----------
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=data['2 Week Actual Price (USD)'],
    y=data['2 Week Prediction (USD)'],
    hue=data['Error'],
    palette="coolwarm"
)
plt.title("Predicted vs Actual Prices", fontsize=16)
plt.xlabel("Actual Price (USD)")
plt.ylabel("Predicted Price (USD)")
plt.tight_layout()
plt.show()


# ---------- 5. Direction Accuracy Bar Chart ----------
plt.figure(figsize=(7,6))
sns.barplot(
    x=['Correct Direction', 'Incorrect Direction'],
    y=[data['Correct Direction'].sum(), len(data) - data['Correct Direction'].sum()],
    palette="Set2"
)
plt.title("Direction Prediction Accuracy", fontsize=16)
plt.ylabel("Count of Stocks")
plt.tight_layout()
plt.show()


# ---------- 6. Within ±5% Accuracy Bar Chart ----------
plt.figure(figsize=(7,6))
sns.barplot(
    x=['Within ±5%', 'Outside ±5%'],
    y=[data['Within 5%'].sum(), len(data) - data['Within 5%'].sum()],
    palette="Set3"
)
plt.title("Percentage of Predictions Within ±5% of Actual Price", fontsize=16)
plt.ylabel("Count of Stocks")
plt.tight_layout()
plt.show()


# ---------- 7. Significant Error (>10%) ----------
plt.figure(figsize=(7,6))
sns.barplot(
    x=['Significant (>10%)', 'Not Significant'],
    y=[data['Significant Difference'].sum(), len(data) - data['Significant Difference'].sum()],
    palette="flare"
)
plt.title("Number of Predictions With >10% Error", fontsize=16)
plt.ylabel("Count of Stocks")
plt.tight_layout()
plt.show()
