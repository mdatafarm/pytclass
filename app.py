import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predictor de Especies de Pingüinos",
    page_icon="🐧",
    layout="centered"
)

# --- Cargar Modelo y Scaler ---
try:
    model = joblib.load('modelo_mlp.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error al cargar los modelos o el escalador: {e}")
    st.stop()

# --- Título y Descripción ---
st.title('🐧 Predictor de Especies de Pingüinos')
st.write("""
Esta aplicación utiliza un modelo de Red Neuronal para predecir la especie de un pingüino
(Adelie, Chinstrap o Gentoo) basándose en sus características físicas.
""")
# st.divider()

# --- Sidebar para la Entrada de Datos del Usuario ---
st.sidebar.header('Introduce las Características del Pingüino:')

def user_input_features():
    """
    Función para recoger los inputs del usuario desde la sidebar.
    """
    island = st.sidebar.selectbox('Isla', ('Dream', 'Torgersen', 'Biscoe'))
    sex = st.sidebar.selectbox('Sexo', ('male', 'female'))
    bill_length_mm = st.sidebar.slider('Largo del Pico (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.sidebar.slider('Profundidad del Pico (mm)', 13.1, 21.5, 17.2)
    body_mass_g = st.sidebar.slider('Masa Corporal (g)', 2700.0, 6300.0, 4201.7)
    year = st.sidebar.slider('Año', 2007, 2009, 2007)

    data = {
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'body_mass_g': body_mass_g,
        'year': year,
        'island_Dream': 1 if island == 'Dream' else 0,
        'island_Torgersen': 1 if island == 'Torgersen' else 0,
        'sex_male': 1 if sex == 'male' else 0,
    }
    
    # Aseguramos el orden correcto de las columnas que el modelo espera
    feature_order = ['bill_length_mm', 'bill_depth_mm', 'body_mass_g', 'year', 
                     'island_Dream', 'island_Torgersen', 'sex_male']
    
    features = pd.DataFrame(data, index=[0])
    features = features[feature_order] # Reordenamos las columnas
    
    return features

df_input = user_input_features()

# --- Mostrar Inputs y Realizar Predicción ---
st.header('Características Seleccionadas (ya procesadas):')
st.write(df_input)

if st.button('¡Predecir Especie!'):
    try:
        # Escalar los datos de entrada usando el scaler cargado
        scaled_features = scaler.transform(df_input)
        
        # Realizar la predicción con los datos escalados
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)

        st.subheader('Predicción:')
        
        # Mapeo de la predicción numérica a la especie (ajusta si tu mapeo es diferente)
        species_map = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
        predicted_species = species_map.get(prediction[0], 'Desconocida')
        
        st.success(f'La especie predicha es **{predicted_species}**.')

        st.subheader('Probabilidad de la Predicción (%):')
        # Creamos un DataFrame legible con las probabilidades
        df_proba = pd.DataFrame(prediction_proba * 100, 
                                columns=['Adelie', 'Chinstrap', 'Gentoo'], 
                                index=['Probabilidad'])
        st.dataframe(df_proba.style.format("{:.2f}%"))
        
    except Exception as e:
        st.error(f"Ocurrió un error al predecir: {e}")