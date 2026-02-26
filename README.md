# 📊 Time Series Data Analysis and Forecasting  
## Task 5 – Machine Learning Forecasting Project  

---

## 📌 Project Overview

This project performs time series data analysis and forecasting using a real-world dataset: **Apple Global Sales Dataset**.

The objective is to analyze historical sales revenue over time and build a forecasting model to predict future revenue using machine learning techniques.

The project covers:

- Time series preprocessing
- Missing value handling
- Trend and seasonal analysis
- Rolling statistics
- Feature engineering
- Forecasting using Linear Regression
- Model evaluation using RMSE and MAE

---

## 📂 Dataset Description

**Dataset Name:** apple_global_sales_dataset.csv  

The dataset contains transaction-level Apple product sales across different regions and dates.

Important columns used:

- `sale_date` → Date of sale
- `revenue_usd` → Revenue generated from each sale

Since the dataset is transactional, it was aggregated to obtain:

> **Total Daily Revenue**

This aggregation created a proper time series dataset suitable for forecasting.

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading & Time Handling

- Loaded dataset using Pandas
- Converted `sale_date` to datetime format
- Aggregated revenue by date
- Set date as index
- Sorted data chronologically

---

### 2️⃣ Missing Value Handling

- Checked for missing values
- Applied **linear interpolation** to handle missing data

**Justification:**  
Linear interpolation preserves time continuity and is appropriate for numeric time series data.

---

### 3️⃣ Exploratory Time Series Analysis

A line plot was created to visualize:

- Overall trend in revenue
- Short-term fluctuations
- General pattern over time

Observations:
- Revenue shows variation over time.
- Short-term fluctuations are visible.
- A long-term trend can be observed.

---

### 4️⃣ Rolling Statistics

Calculated:

- 7-day rolling average
- 30-day rolling average

Purpose:
- 7-day average smooths short-term noise
- 30-day average highlights long-term trend

These rolling averages help understand data smoothing effects.

---

### 5️⃣ Seasonal Decomposition

Performed additive seasonal decomposition to separate:

- Trend component
- Seasonal component
- Residual component

This helped identify repeating patterns and underlying trends in the data.

---

### 6️⃣ Feature Engineering

Created lag features to help the model learn from historical values:

- `lag_1` → Previous day revenue (t−1)
- `lag_7` → Previous week revenue (t−7)

Lag features are essential in time series forecasting.

---

### 7️⃣ Train–Test Split

- 80% data used for training
- 20% data used for testing
- Chronological order maintained
- No random shuffling applied

---

### 8️⃣ Forecasting Model

Model Used: **Linear Regression**

Reason for Selection:
- Simple and interpretable
- Suitable for baseline time series forecasting
- Works effectively with lag-based features

---

### 9️⃣ Model Evaluation

Model performance was evaluated using:

- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

Lower RMSE and MAE values indicate better predictive performance.

---

## 📈 Results

The model was able to capture general revenue patterns using lag features.

- RMSE: (value from output)
- MAE: (value from output)

The predicted values follow the actual trend reasonably well, though some deviations exist due to model simplicity.

---

## 📊 Tools & Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- Scikit-learn

---

## 🚀 Conclusion

This project demonstrates how time series data can be:

- Cleaned and preprocessed
- Analyzed for trends and seasonality
- Enhanced using lag-based feature engineering
- Forecasted using machine learning models

While Linear Regression provides a good baseline, future improvements may include:

- ARIMA or SARIMA models
- Prophet
- LSTM (Deep Learning)
- Hyperparameter tuning
- Additional lag features

---

## 📌 Project Status

✅ Completed successfully  
✅ All assignment requirements satisfied  
