import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

df.info()

df = pd.read_csv("powerconsumption.csv")
total_power = df[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]
df['Total_Power_Sum'] = total_power.sum(axis=1)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
df["Month"] = df["Datetime"].dt.month
df_1_to_11 = df[df['Month'].isin(range(1, 12))]
df_12 = df[df['Month'] == 12]
df_1_to_11.to_csv("1to11.csv", index=False)
df_12.to_csv("to12.csv", index=False)
print(df_1_to_11.head())
print(df_12.head())

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
total_power = df_1_to_11[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]
df_1_to_11['Total_Power_Sum'] = total_power.sum(axis=1)
df_1_to_11 = df_1_to_11.drop('Datetime', axis = 1)
df_1_to_11 = df_1_to_11.drop('PowerConsumption_Zone1', axis = 1)
df_1_to_11 = df_1_to_11.drop('PowerConsumption_Zone2', axis = 1)
df_1_to_11 = df_1_to_11.drop('PowerConsumption_Zone3', axis = 1)
train_df, test_df = train_test_split(df_1_to_11, test_size=0.2, random_state=42)
X_train = train_df.drop("Total_Power_Sum", axis=1).values.reshape(-1, 1, 6)
X_test = test_df.drop("Total_Power_Sum", axis=1).values.reshape(-1, 1, 6)
y_train = train_df["Total_Power_Sum"].values.reshape(-1, 1)
y_test = test_df["Total_Power_Sum"].values.reshape(-1, 1)

model=Sequential()
model.add(LSTM(256,return_sequences=True,input_shape=(1, 6)))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

model = Sequential()
model.add(LSTM(32, input_shape=(1, 6)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(X_train, y_train, epochs=20, verbose=1)

preds = model.predict(X_test)
df_12 = df_12.drop('Datetime', axis = 1)
df_12 = df_12.drop('PowerConsumption_Zone1', axis = 1)
df_12 = df_12.drop('PowerConsumption_Zone2', axis = 1)
df_12 = df_12.drop('PowerConsumption_Zone3', axis = 1)

next_24_hours_X = df_12[["Temperature", "Humidity", "WindSpeed","GeneralDiffuseFlows","DiffuseFlows", "Month"]].values.reshape(-1, 1, 6)
next_24_hours_preds = model.predict(next_24_hours_X)
df_12["Total_Power_Sum"] = next_24_hours_preds.flatten()
df_12.to_csv("PredictData.csv", index=False)

data = pd.read_csv("to12.csv")
datapredict = pd.read_csv("PredictData.csv")
mapedf = np.mean(np.abs((data["Total_Power_Sum"] - datapredict["Total_Power_Sum"]) / data["Total_Power_Sum"])) * 100
mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

import numpy as np
data = pd.read_csv("to12.csv")
datapredict = pd.read_csv("PredictData.csv")
mapedf = np.mean(np.abs((data["Total_Power_Sum"] - datapredict["Total_Power_Sum"]) / data["Total_Power_Sum"] )) * 100
mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1))) * 100  # заменяем деление на ноль значением нуля
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

datapredict
data
