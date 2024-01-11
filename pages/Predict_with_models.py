import streamlit as st
import pandas as pd
import pickle
import numpy as np


uploaded_file = st.file_uploader("Выберите файл датасета")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Загруженный датасет:", df)

    st.title("Датасет card_transdata")

st.title("Получить предсказание мошенничества.")
st.write("Далее обозначим: 1 - правда, 0 - ложь")

st.header("Дистанция от дома")
distance_from_home = st.number_input("Число:", min_value=0, max_value=10000, value=4)

st.header("Дистанция от прошлой транзакции")
distance_from_tranzaction = st.number_input("Число:", min_value=0, max_value=10000, value=1195)

st.header("Отношение суммы покупки к медианной цене покупки")
amount_per_median = st.number_input("Число:", min_value=0, max_value=10000, value=131)

st.header("Произошла ли транзакция от того же продавца")
retailer = [0,1]

retailerMeaning = st.selectbox("Правда?", retailer)

st.header("Происходит ли транзакция через чип (кредитную карту)")

chipMeaning = st.selectbox("Правда? ", [0,1])

st.header("Произошла ли транзакция с использованием PIN-кода")

pinMeaning = st.selectbox("Правда?  ", retailer)

st.header("Является ли транзакция онлайн-заказом")

orderMeaning = st.selectbox("Правда?   ", retailer)



data = pd.DataFrame({'distance_from_home': [distance_from_home],
                    'distance_from_last_transaction': [distance_from_tranzaction],
                    'ratio_to_median_purchase_price': [amount_per_median],
                    'repeat_retailer': [retailerMeaning],
                    'used_chip': [chipMeaning],
                    'used_pin_number': [pinMeaning],
                    'online_order': [orderMeaning]})


button_clicked = st.button("Предсказать")

if button_clicked:
    st.header("1 - мошенничество")
    st.header("0 - всё ок")
    with open('..\models/KNN.pickle', 'rb') as file:
        knn_model = pickle.load(file)
    with open('..\models/Bagging.pickle', 'rb') as file:
        bagging_model = pickle.load(file)
    with open('..\models/Gradient.pickle', 'rb') as file:
        gradient_model = pickle.load(file)
    with open('..\models/KMeans.pickle', 'rb') as file:
        kmeans_model = pickle.load(file)
    from tensorflow.keras.models import load_model
    nn_model = load_model('..\models/nn/nn.h5')

    with open('..\models/Stacking.pickle', 'rb') as file:
        stacking_model = pickle.load(file)

    st.header("KNN говорит:")
    pred =[]
    knn_pred = knn_model.predict(data)[0]
    pred.append(int(knn_pred))
    st.write(f"{knn_pred}")

    st.header("bagging говорит:")
    bagging_pred = bagging_model.predict(data)[0]
    pred.append(int(bagging_pred))
    st.write(f"{bagging_pred}")

    st.header("gradient говорит:")
    gradient_pred = gradient_model.predict(data)[0]
    pred.append(int(gradient_pred))
    st.write(f"{gradient_pred}")

    st.header("kmeans говорит:")
    kmeans_pred = kmeans_model.predict(data)[0]
    pred.append(int(kmeans_pred))
    st.write(f"{kmeans_pred}")

    st.header("NN говорит:")
    nn_pred = round(nn_model.predict(data)[0][0])
    pred.append(nn_pred)
    st.write(f"{nn_pred}")

    st.header("Stacking говорит:")
    stacking_pred = stacking_model.predict(data)[0]
    pred.append(int(stacking_pred))
    st.write(f"{stacking_model.predict(data)[0]}")

    st.header("Итоговый ответ:")
    answer = {0:"Всё норм",
              1:"Мошенничество"}
    st.write(answer[np.argmax(np.bincount(pred))])
