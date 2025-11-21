import pandas as pd

# Load the dataset
data = pd.read_csv("stockdata.csv")
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Clean column names
data.columns = data.columns.str.strip()

# Calculate changes
data['Predicted Change'] = data['1 Month Prediction (USD)'] - data['Price on 12/17/2024 (USD)']
data['Actual Change'] = data['1 Month Actual Price (USD)'] - data['Price on 12/17/2024 (USD)']

# Determine prediction direction
data['Predicted Up'] = data['Predicted Change'] > 0
data['Actual Up'] = data['Actual Change'] > 0
data['Correct Direction'] = data['Predicted Up'] == data['Actual Up']

# Calculate within 5% range
data['Within 5%'] = abs(data['1 Month Prediction (USD)'] - data['1 Month Actual Price (USD)']) / data['1 Month Actual Price (USD)'] <= 0.05

# Calculate statistical significance (e.g., >10% difference)
data['Significant Difference'] = abs(data['1 Month Prediction (USD)'] - data['1 Month Actual Price (USD)']) / data['1 Month Actual Price (USD)'] > 0.1

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
print(predicted_up_and_went_up[['Ticker', 'Price on 12/17/2024 (USD)', '1 Month Prediction (USD)', '1 Month Actual Price (USD)']])

print("\nStocks that were predicted to go down and actually went down:")
print(predicted_down_and_went_down[['Ticker', 'Price on 12/17/2024 (USD)', '1 Month Prediction (USD)', '1 Month Actual Price (USD)']])

# print("\n\nStocks that were predicted to go up and actually went up:")
# print(predicted_up_and_went_up['Ticker'])
# print("\nStocks that were predicted to go down and actually went down:")
# print(predicted_down_and_went_down['Ticker'])