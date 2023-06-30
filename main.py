import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#from mpl_toolkits.mplot3d import Axes3D
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

df = pd.read_csv("powerconsumption.csv")

df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

print("Набор данных: ",df.shape)

print("Дубликаты: ",df.duplicated().sum())

print("Пропущенные значения: ",df.isnull().sum())

#Создание нового столбца в наборе данных
total_power = df[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]
df['Total_Power_Sum'] = total_power.sum(axis=1)
df["Month"] = df["Datetime"].dt.month

#Установка индекса на столбец Datetime
df = df.set_index(df["Datetime"])

#Группировка данных по месяцам для более простого и эффективного построения трендов.
grouped = df.groupby('Month').mean(numeric_only=True)

fig = px.line(df,
              x="Datetime",
              y="Humidity",
              labels = {'Datetime':'Months'},
              title = "Общая влажность за год")
#fig.show()

fig = px.line(df,
              x="Datetime",
              y="Total_Power_Sum",
              labels = {'Datetime':'Months'},
              title = "Общая мощность, выработанная за год")

fig.update_layout(
    template='plotly',
    font=dict(size=10),
    title={
        'text': "Общая мощность, выработанная за год",
        'font': {'size': 34}
    }
)
#fig.show()

fig = px.box(df,
        x=df.index.month,
        y="Total_Power_Sum",
        color=df.index.month,
        labels = {"x" : "Месяцы"},
        title="Выработка электроэнергии | Месячная статистика ")

fig.update_traces(width=0.5)
#fig.show()

fig = px.box(df,
        x=df.index.day,
        y="Total_Power_Sum",
        color=df.index.day,
        labels = {"x" : "Дни"})

fig.update_traces(width=0.5)
#fig.show()

fig = px.bar(grouped,
              x=grouped.index,
              y="Total_Power_Sum",
              labels = {'Month':'Месяцы'},
              color = "Total_Power_Sum",
              title="Выработка электроэнергии в месяц")
fig.update_traces(width=0.6)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
#fig.show()

fig = px.bar(grouped,
              x = grouped.index,
              y = "WindSpeed",
              labels = {'Month':'Месяцы'},
              color = "WindSpeed",
              title="Скорость ветра в месяц")
fig.update_traces(width=0.6)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
#fig.show()

fig = px.bar(grouped,
              x=grouped.index,
              y="Temperature",
              labels = {'Month':'Month'},
              color = "Temperature",
              title="Средняя температура воздуха в месяц")
fig.update_traces(width=0.6)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
#fig.show()

fig = px.box(df,
             y="Total_Power_Sum",
             title="Общая статистика выработки электроэнергии")

#fig.show()

fig = px.box(df,
             y="WindSpeed",
             title="Общая статистика скорости ветра")

#fig.show()

df_corr = df.corr()

x = list(df_corr.columns)
y = list(df_corr.index)
z = np.array(df_corr)

fig = ff.create_annotated_heatmap(x = x,
                                  y = y,
                                  z = z,
                                  annotation_text = np.around(z, decimals=2))
#fig.show()

fig, ax = plt.subplots(figsize = (20,15))
plt.hist(df['Total_Power_Sum'])
plt.title('Распределение выработки', fontsize = 24)
ax.set_xlabel('Все что угодно', fontsize = 24)
ax.set_ylabel('y' , fontsize = 24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.show()

fig, ax = plt.subplots(figsize = (20,15))
plt.hist(df['WindSpeed'])
plt.title('Распределение скорости ветра', fontsize = 24)
ax.set_xlabel('x', fontsize = 24)
ax.set_ylabel('y' , fontsize = 24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.show()

# Разделить датасет на 2 датафрейма
df = pd.read_csv("powerconsumption.csv")
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
df["Month"] = df["Datetime"].dt.month

df_1_to_11 = df[df['Month'].isin(range(1, 12))] # Выбрать строки с месяцами от 1 до 11
df_12 = df[df['Month'] == 12] # Выберать строки с месяцем 12

# Распечатайте первые 5 строк каждого фрейма данных
df_1_to_11.to_csv("1to11.csv", index=False)
df_12.to_csv("to12.csv", index=False)
#print(df_1_to_11.head())
#print(df_12.head())
"""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

total_power = df[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]
df_1_to_11['Total_Power_Sum'] = total_power.sum(axis=1)

df_1_to_11 = df_1_to_11.drop('Datetime', axis = 1)

# Разделить датасеты на обучащиен и тестоввые сеты?
train_df, test_df = train_test_split(df_1_to_11, test_size=0.2, random_state=42)

# Преобразуйте данные обучения и тестирования в 3д массив для ввода LSTM.
X_train = train_df.drop("Total_Power_Sum", axis=1).values.reshape(-1, 1, 9)
X_test = test_df.drop("Total_Power_Sum", axis=1).values.reshape(-1, 1, 9)
y_train = train_df["Total_Power_Sum"].values.reshape(-1, 1)
y_test = test_df["Total_Power_Sum"].values.reshape(-1, 1)

# Создаем LSTM модель
model=Sequential()
model.add(LSTM(256,return_sequences=True,input_shape=(1, 9)))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

# Создаем LSTM модель
model = Sequential()
model.add(LSTM(32, input_shape=(1, 9)))  #32
model.add(Dense(64, activation='relu'))  #64 3.3%
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# Обучаем модель LSTM
model.fit(X_train, y_train, epochs=1, verbose=0)

preds = model.predict(X_test)

next_24_hours_X = df_12[["Temperature", "Humidity", "WindSpeed","Month"]].values.reshape(-1, 1, 9)
next_24_hours_preds_extended = np.concatenate([next_24_hours_preds] * 3 , axis=0)#

# Делайте прогнозы, используя модель
next_24_hours_preds = model.predict(next_24_hours_X)
# Добавляем прогнозируемые значения в исходный датафрейм
df_12["Total_Power_Sum"] = next_24_hours_preds_extended.flatten()[:4320]
# Запишите обновленные данные в тот же файл Excel
df_12.to_csv("PredictData.csv", index=False)

data = pd.read_csv("to12.csv")
total_power = data[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]
data['Total_Power_Sum'] = total_power.sum(axis=1)

datapredict = pd.read_csv("PredictData.csv")
mapedf = np.mean(np.abs((data["Total_Power_Sum"] - datapredict["Total_Power_Sum"]) / data["Total_Power_Sum"])) * 100
mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

print("Метрики работы модели:")
print("__________________________________________________________________")
print("Model Percentage Mean Absolute Error: ", mape)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
print("R^2: ", r2)
print("Percentage Mean Absolute Error: ", mapedf)
print("__________________________________________________________________")

import numpy as np
data = pd.read_csv("to12.csv")
total_power = data[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]
data['Total_Power_Sum'] = total_power.sum(axis=1)
datapredict = pd.read_csv("PredictData.csv")
# Calculate the model performance metrics

mapedf = np.mean(np.abs((data["Total_Power_Sum"] - datapredict["Total_Power_Sum"]) / data["Total_Power_Sum"] )) * 100
mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1))) * 100  # заменяем деление на ноль значением нуля
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

# ВыВОДИМ показатели производительности модели
print("Metrics of model performance:")
print("__________________________________________________________________")
print("Model Percentage Mean Absolute Error: ", mape)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
print("R^2: ", r2)
print("Percentage Mean Absolute Error: ", mapedf)
print("__________________________________________________________________")
"""