import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle


path_model = "linear_regression.pickle"

model_file = open(path_model,"rb")
model = pickle.load(model_file)
model_file.close()


st.set_page_config(page_title="ML Regresión Lineal: Minimos cuadrados",layout="centered")

st.title("Modelo de Machine Learning")
st.subheader("Regresión Lineal: Mínimos cuadrados")
steps = st.tabs(["Funcionamiento","Predecir IPC anual","Ver evolución"])

with steps[0]:
    st.image("lr_theory.png")

with steps[1]:
    st.markdown("## Predicción del indice de precios al consumidor")
    year = st.number_input("Escoja un año", min_value=2026)

    year_to_predict = np.array([year]).reshape(-1,1)
    valor_ipc_pred = model.predict(year_to_predict).tolist()
    valor_ipc_pred = valor_ipc_pred[0][0]
    st.button("Calcular IPC", key="calcular")

    if st.session_state["calcular"]: st.write(f"El IPC predicho para el año {year} es {valor_ipc_pred:.2f}")

with steps[2]:

    scatter_data = pd.read_csv('scatter_data.csv')
    years = scatter_data['x'].unique().reshape(-1,1)

    x = scatter_data['x']
    y = scatter_data['y']

    y_pred = model.predict(years)

    
    fig, ax = plt.subplots()
    ax = plt.scatter(x, y, color="#ff2186", sizes=[2.5])
    ax = plt.plot(years, y_pred)
    st.pyplot(fig)

    pass
