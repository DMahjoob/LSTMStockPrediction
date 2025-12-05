# 📈 Stock Price Prediction App

*A Flask-powered LSTM stock forecasting tool with interactive visualizations.*

---

## 🚀 Overview

This project provides a web-based application for forecasting stock prices using a PyTorch LSTM model.
Users can enter a stock ticker, choose a forecast horizon (1 month, 6 months, or 1 year), and receive:

* A predicted price for the selected period
* A full future price curve
* An auto-generated plot displayed directly in the browser

The repository also includes standalone analytics tools for evaluating prediction accuracy across historical datasets.

---

### 🔮 **Web App (Flask)**

* Clean UI to input ticker and prediction period
* Pulls 10 years of stock data using **Yahoo Finance**
* Trains an LSTM model on-the-fly using PyTorch
* Produces multi-day forward predictions
* Generates and embeds a matplotlib plot dynamically
* Caching system avoids repeated downloads from Yahoo Finance

### 📊 **StockAnalytics.py (Data Science Module)**

Includes:

* Statistical comparison of predicted vs actual stock behavior
* Direction accuracy (up/down)
* Error distribution visualizations
* Residual histograms, scatterplots, and bar charts
* Support for entire CSV-based datasets

---

## 🗂 Repository Structure

```
static/
  styles.css
templates/
  index.html
  result.html
app.py
README.md
requirements.txt
StockAnalytics.py
stocks.csv
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Web App

Run the Flask server:

```bash
python app.py
```

The app will start on:

```
http://0.0.0.0:8000
```

You can now enter a stock ticker (e.g., AAPL, TSLA) and generate predictions.

---

## 🧩 Tech Stack

**Frontend:** HTML, CSS
**Backend:** Flask (Python)
**Machine Learning:** PyTorch LSTM
**Data:** Yahoo Finance API (via `yfinance`)
**Visualization:** Matplotlib
**Data Analysis:** Pandas, NumPy, Seaborn
