import streamlit as st

st.title("Информация о датасете")

st.header("Тематика датасета")
st.write("Мошенничество с кредитными картами")

st.header("Объяснение функции:")
st.write("distance_from_home — расстояние от дома, где произошла транзакция.")
st.write("distance_from_last_transaction — расстояние от последней транзакции.")
st.write("Ratio_to_median_purchase_price — отношение суммы покупки к медианной цене покупки")
st.write("repeat_retailer — произошла ли транзакция от того же продавца.")
st.write("Used_chip - Происходит ли транзакция через чип (кредитную карту).")
st.write("Used_pin_number — произошла ли транзакция с использованием PIN-кода.")
st.write("online_order — является ли транзакция онлайн-заказом.")
st.write("fraud – является ли транзакция мошеннической.")

st.header("Особенности предобработки данных")
st.write("В датасете необходимо предугатыватьявляется ли транзакция мошеннической. Мошенничество опеределяется показателем 1 или 0.")
st.write("1 - мошенничество")
st.write("0 - обычная операция")
st.write("В датасете отсутствовали пропущенные значения и дубликаты.")

st.write("Был проведен EDA (см. Vusualization) и определены коррелирующиеся признаки")
st.write("Числовые признаки были масштабированны. Дисбаланс был устраннен алгоритмом NearMiss")