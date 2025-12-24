import streamlit as st
import pandas as pd
import joblib
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title="Shoppers Analysis", layout="wide")

st.title("Анализ выбора покупателей")

@st.cache_data
def load_data():
    return pd.read_csv('data/online_shoppers_intention.csv')

df = load_data()

tab1, tab2 = st.tabs(["Дашборд", "Модель"])

with tab1:
    st.header("Анализ данных")
    
    if st.button("Показать отчёт"):
        with st.spinner('Генерация отчёта...'):
            profile = ProfileReport(df, explorative=True)
            st_profile_report(profile)

with tab2:
    st.header("Проверка модели")
    
    model = joblib.load('models/model.pkl')
    st.success("Модель загружена")
        
    if st.button("Проверить на случайном примере"):
        df_clean = pd.read_csv('data/df_clean.csv')
        
        sample = df_clean.sample(1)
        
        true_val = sample['Revenue'].values[0]
        X_sample = sample.drop('Revenue', axis=1)
        
        prediction = model.predict(X_sample)[0]
        proba = model.predict_proba(X_sample)[0]
        
        st.write("Данные пользователя:")
        st.dataframe(X_sample)
        
        col1, col2 = st.columns(2)
        
        result_text = "Купит" if prediction == 1 else "Не купит"
        col1.metric("Прогноз модели", result_text)
        col1.write(f"Вероятность: {proba[1]:.2%}")
        
        real_text = "Купил" if true_val == 1 else "Не купил"
        col2.metric("На самом деле", real_text)