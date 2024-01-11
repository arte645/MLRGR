import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('..\pythonProject3/venv/data/card_transdata.csv')

st.title("Датасет card_transdata")

st.header("Тепловая карта с корреляцией между основными признаками")

plt.figure(figsize=(12, 8))
selected_cols = ['fraud', 'ratio_to_median_purchase_price', 'used_pin_number', 'online_order']
selected_df = df[selected_cols]
sns.heatmap(selected_df.corr(), annot=True, cmap='coolwarm')
plt.title('Тепловая карта с корреляцией')
st.pyplot(plt)

st.header("Гистограмма distance_from_home")

columns = ['distance_from_home']

for col in columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df.sample(5000)[col], bins=100, kde=True)
    plt.title(f'Гистограмма для {col}')
    st.pyplot(plt)

st.header("Ящик с усами для distance_from_home")
outlier = df[columns]
Q1 = outlier.quantile(0.25)
Q3 = outlier.quantile(0.75)
IQR = Q3-Q1
data_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) |(outlier > (Q3 + 1.5 * IQR))).any(axis=1)]


for col in columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data_filtered[col])
    plt.title(f'{col}')
    plt.xlabel('Значение')
    st.pyplot(plt)

st.header("Круговая диаграмма целевого признака")
plt.figure(figsize=(8, 8))
df['fraud'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('fraud')
plt.ylabel('')
st.pyplot(plt)
