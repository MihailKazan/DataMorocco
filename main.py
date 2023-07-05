import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#from mpl_toolkits.mplot3d import Axes3D
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

#Импорт набора данных
df = pd.read_csv("powerconsumption.csv")

#Просмотр верхней части данных
df.head()

#Просмотр нижней части данных
df.tail()

#Просмотр верхней и нижней части данных
df

#Просмотр состава набора данных
df.shape

#Просмотр типа данных и проверка на нулевые значения
df.info()

df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

df

df.info()

#Проверка на пропущенные значения
df.isnull().sum()

#Проверка на дубликаты в данных
df.duplicated().sum()

#Просмотр описательной статистики в наборе данных
df.describe()

#Создание нового столбца в наборе данных
total_power = df[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]
df['Total_Power_Sum'] = total_power.sum(axis=1)
df["Month"] = df["Datetime"].dt.month
df

#Просмотр описательной статистики в наборе данных с округлением до 2х знаков после запятой
df.describe().round(2)

#Установка индекса на столбец Datetime
df = df.set_index(df["Datetime"])
df

#Группировка данных по месяцам для более простого и эффективного построения трендов.
grouped = df.groupby('Month').mean(numeric_only=True)
grouped

#Общая влажность
fig = px.line(df,
              x="Datetime",
              y="Humidity",
              labels = {'Datetime':'Months'},
              title = "Общая влажность за год")
fig.show()

#Общая мощность выработанная за год
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
fig.show()

#Выработка энергии(месячная статистика)
fig = px.box(df,
        x=df.index.month,
        y="Total_Power_Sum",
        color=df.index.month,
        labels = {"x" : "Месяцы"},
        title="Выработка электроэнергии | Месячная статистика ")
fig.update_traces(width=0.5)
fig.show()

fig = px.box(df,
        x=df.index.day,
        y="Total_Power_Sum",
        color=df.index.day,
        labels = {"x" : "Дни"})
fig.update_traces(width=0.5)
fig.show()

fig = px.bar(grouped,
              x=grouped.index,
              y="Total_Power_Sum",
              labels = {'Month':'Месяцы'},
              color = "Total_Power_Sum",
              title="Выработка электроэнергии в месяц")
fig.update_traces(width=0.6)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

fig = px.bar(grouped,
              x = grouped.index,
              y = "WindSpeed",
              labels = {'Month':'Месяцы'},
              color = "WindSpeed",
              title="Скорость ветра в месяц")
fig.update_traces(width=0.6)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

#Средняя температура  в месяц
fig = px.bar(grouped,
              x=grouped.index,
              y="Temperature",
              labels = {'Month':'Month'},
              color = "Temperature",
              title="Средняя температура воздуха в месяц")
fig.update_traces(width=0.6)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

fig = px.box(df,
             y="Total_Power_Sum",
             title="Общая статистика выработки электроэнергии")
fig.show()

fig = px.box(df,
             y="WindSpeed",
             title="Общая статистика скорости ветра")
fig.show()

#Делаем корреляцию
df_corr = df.corr()
df_corr

x = list(df_corr.columns)
y = list(df_corr.index)
z = np.array(df_corr)
fig = ff.create_annotated_heatmap(x = x,
                                  y = y,
                                  z = z,
                                  annotation_text = np.around(z, decimals=2))
fig.show()

fig, ax = plt.subplots(figsize = (20,15))
plt.hist(df['Total_Power_Sum'])
plt.title('Распределение выработки', fontsize = 24)
ax.set_xlabel('Все что угодно', fontsize = 24)
ax.set_ylabel('y' , fontsize = 24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

fig, ax = plt.subplots(figsize = (20,15))
plt.hist(df['WindSpeed'])
plt.title('Распределение скорости ветра', fontsize = 24)
ax.set_xlabel('x', fontsize = 24)
ax.set_ylabel('y' , fontsize = 24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()