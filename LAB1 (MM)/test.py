import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("LAB1 (MM)/saveecobot_22643.csv")
#print(data.columns)
data_wide = data.pivot_table(index="logged_at", columns="phenomenon", values="value").reset_index()

data_wide["logged_at"]=pd.to_datetime(data_wide["logged_at"])
data_wide["hour"]=data_wide["logged_at"].dt.hour

#print(data_wide.head(10))

data_clean = data_wide.dropna()

print("Отримання значень одного забрудника (рм2.5) від інших (no2_ug, so2_ug, co_mg, no2_ppb, so2_ppb, co_ppm)")

X = data_clean[["pm1", "pm10"]]
Y = data_clean["pm25"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=21)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)



print("MSE: ", mse)
print("R2: ", r2)

print("Залежність забрудника від часу дня (pm2.5)")

X_time = data_clean[["hour"]]
Y_time = data_clean["pm25"]

#print(X_time)

X_time_train, X_time_test, Y_time_train, Y_time_test = train_test_split(X_time, Y_time, test_size=0.2, random_state=21)

model_time = LinearRegression()
model_time.fit(X_time_train, Y_time_train)

Y_time_pred = model_time.predict((X_time_test))

mse_time = mean_squared_error(Y_time_test, Y_time_pred)
r2_time = r2_score(Y_time_test, Y_time_pred)

print("MSE time: ", mse_time)
print("R2 time: ", r2_time)