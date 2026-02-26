# ==========================================
# TASK 5: Time Series Forecasting
# Dataset: apple_global_sales_dataset.csv
# Forecasting: Daily Total Revenue
# ==========================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 2️⃣ Load Dataset
# ==========================================

df = pd.read_csv("apple_global_sales_dataset.csv")

# Convert sale_date to datetime
df['sale_date'] = pd.to_datetime(df['sale_date'])

# ==========================================
# 3️⃣ Aggregate to Daily Revenue (IMPORTANT STEP)
# ==========================================

daily_data = df.groupby('sale_date')['revenue_usd'].sum().reset_index()

daily_data.set_index('sale_date', inplace=True)
daily_data = daily_data.sort_index()

print("First 5 rows of Daily Aggregated Data:")
print(daily_data.head())

# ==========================================
# 4️⃣ Missing Value Handling
# ==========================================

print("\nMissing Values Before Handling:")
print(daily_data.isnull().sum())

# Interpolation (safe for time series)
daily_data['revenue_usd'] = daily_data['revenue_usd'].interpolate(method='linear')

print("\nMissing Values After Handling:")
print(daily_data.isnull().sum())

# ==========================================
# 5️⃣ Exploratory Time Series Plot
# ==========================================

plt.figure(figsize=(12,5))
plt.plot(daily_data['revenue_usd'])
plt.title("Daily Total Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue (USD)")
plt.show()

# ==========================================
# 6️⃣ Rolling Statistics
# ==========================================

daily_data['Rolling_7'] = daily_data['revenue_usd'].rolling(7).mean()
daily_data['Rolling_30'] = daily_data['revenue_usd'].rolling(30).mean()

plt.figure(figsize=(12,5))
plt.plot(daily_data['revenue_usd'], label='Original')
plt.plot(daily_data['Rolling_7'], label='7-Day Avg')
plt.plot(daily_data['Rolling_30'], label='30-Day Avg')
plt.legend()
plt.title("Rolling Averages")
plt.show()

# ==========================================
# 7️⃣ Seasonal Decomposition
# ==========================================

decomposition = seasonal_decompose(daily_data['revenue_usd'], model='additive', period=7)
decomposition.plot()
plt.show()

# ==========================================
# 8️⃣ Feature Engineering (Lag Features)
# ==========================================

daily_data['lag_1'] = daily_data['revenue_usd'].shift(1)
daily_data['lag_7'] = daily_data['revenue_usd'].shift(7)

daily_data.dropna(inplace=True)

# ==========================================
# 9️⃣ Train-Test Split (80% Train, 20% Test)
# ==========================================

train_size = int(len(daily_data) * 0.8)

train = daily_data[:train_size]
test = daily_data[train_size:]

X_train = train[['lag_1', 'lag_7']]
y_train = train['revenue_usd']

X_test = test[['lag_1', 'lag_7']]
y_test = test['revenue_usd']

# ==========================================
# 🔟 Build Linear Regression Model
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# 1️⃣1️⃣ Forecast
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# 1️⃣2️⃣ Model Evaluation
# ==========================================

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print("\nModel Evaluation:")
print("RMSE:", rmse)
print("MAE:", mae)

# ==========================================
# 1️⃣3️⃣ Plot Actual vs Predicted
# ==========================================

plt.figure(figsize=(12,5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, predictions, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Revenue")
plt.show()

print("\n✅ TASK COMPLETED SUCCESSFULLY!")
