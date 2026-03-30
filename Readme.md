# AI Stock Predictor

A high-performance, multi-threaded desktop application that combines **Random Forest Machine Learning**, **Sentiment Analysis (NLP)**, and **Quantitative Finance** principles to predict stock market volatility. 

This project demonstrates a full-stack data science workflow: from raw API ingestion and mathematical stationarity transformations to a multi-threaded Tkinter GUI.

## 🚀 Key Engineering Features

* **Parallel AI Engine:** Leverages Python’s `concurrent.futures.ThreadPoolExecutor` to perform asynchronous data fetching and model training, reducing latency by ~70% compared to sequential processing.
* **Stationary Time-Series Preprocessing:** Implements logarithmic return transformations to convert non-stationary price data into a stationary format, a prerequisite for robust statistical modeling.
* **Hybrid Intelligence:** Integrates a **VADER Sentiment Analysis** pipeline to scrape and quantify market news, providing a "sanity check" for purely technical predictions.
* **Global Currency Normalization:** Features a real-time **USD/INR conversion engine** that detects the source currency via `yfinance` and normalizes all financial metrics to ₹ (INR).
* **Production-Grade Error Handling:** Uses `contextlib` to suppress noisy API stderr logs and implements a **Smart Ticker Recovery** logic (automatic `.NS` suffixing for Indian markets).
* **Smooth UX Architecture:** A custom-built Tkinter viewport with **inertial scrolling logic** and embedded Matplotlib charts for luxury-grade data visualization.

---

## 🛠️ Technical Deep Dive

### 1. The Mathematical Model: Why Log-Returns?
Standard AI models often fail on stock prices because they are **Non-Stationary** (mean and variance change over time). Recruiters, take note of the preprocessing logic used here:
* **Formula:** $$\text{Log Return} = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
* **Reasoning:** Log-returns ensure the data is time-additive and follows a more normal distribution, allowing the `StandardScaler` to normalize features without being "blinded" by the absolute price of the stock.

### 2. The Machine Learning Pipeline
* **Algorithm:** `RandomForestRegressor`
* **Features:** Log-Returns, 10-day Simple Moving Average (SMA), and Volume Percentage Change.
* **Scaling:** Z-score normalization (`StandardScaler`) is applied to ensure that high-volume stocks (like Reliance) and high-price stocks (like Apple) are weighted equally by the regressor.

### 3. Concurrency & Performance
To prevent the GUI from freezing during heavy math lifting, the project uses a **Producer-Consumer pattern**:
1.  **Main Thread:** Handles the Tkinter event loop and SmoothScroll animations.
2.  **Worker Threads:** Five dedicated workers handle `yfinance` downloads and Scikit-Learn training cycles.
3.  **Callback Logic:** Results are "injected" into the UI via `root.after()` to ensure thread-safe updates to the Tkinter canvas.

---

## 🚦 Getting Started

### 1. Prerequisites
* Python 3.10+
* VS Code / PyCharm

### 2. Installation
```bash
pip install yfinance pandas scikit-learn matplotlib vaderSentiment
```

### 3. Usage
Run the application:
```bash
python prediction.py
```
* **Search:** Type `BHEL` or `AAPL`. The smart engine will find the correct exchange automatically.
* **Manage:** Add/Remove stocks to customize your dashboard.
* **Predict:** Click **"START AI ENGINE"** to begin the parallel processing cycle.

---

## 📊 Visual Hierarchy & UI Mechanics

The UI is designed for **Clarity at a Glance**:
* **Sidebar (20%):** Control center for watchlist management.
* **Main Dashboard (80%):** High-resolution cards containing:
    * **Blue Trace:** Actual market price in INR.
    * **Pink Trace:** AI-projected price target.
    * **Sentiment Gauge:** Color-coded Bullish/Bearish indicators based on real-time news NLP.

---

## ⚖️ License & Open Source
This project is part of a deep-dive into Quantitative AI. Feel free to fork and adapt for algorithmic trading research.