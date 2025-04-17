import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Rusty Bargain - Predicción de Autos", layout="wide")

# Carga de datos 
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Datos cargados desde el archivo subido")
        else:
            possible_paths = [
                "datasets/autos.csv",
                "autos.csv",
                "./autos.csv",
                os.path.join("datasets", "autos.csv")
            ]

            # Ruta local
            if os.path.exists("C:\\Users\\ricar\\Desktop\\rusty_bargain\\datasets\\autos.csv"):
                possible_paths.insert(0, "C:\\Users\\ricar\\Desktop\\rusty_bargain\\datasets\\autos.csv")

            df = None
            for path in possible_paths:
                try:
                    df = pd.read_csv(path)
                    st.success(f"Datos cargados correctamente desde: {path}")
                    break
                except FileNotFoundError:
                    continue

            if df is None:
                st.info("No se encontró el archivo de datos. Por favor, carga un archivo CSV.")
                return pd.DataFrame()

        return df

    except Exception as e:
        st.error(f"Ocurrió un error al cargar los datos: {e}")
        return pd.DataFrame()

# Interfaz de carga de CSV
st.sidebar.header("Carga de Datos")
custom_file = st.sidebar.file_uploader("Sube tu propio CSV", type=["csv"])

# Carga de datos 
data = load_data(custom_file)

if not data.empty:
    st.title("Rusty Bargain - Análisis de Valor de Mercado de Autos Usados")

    st.markdown("""
    Esta app permite explorar el dataset y visualizar insights clave del proyecto de ciencia de datos para predecir el valor de reventa de autos usados.
    """)

    st.subheader("Vista previa del dataset")
    st.dataframe(data.head(20))

    st.sidebar.header("Opciones de Exploración")
    selected_col = st.sidebar.selectbox("Selecciona una variable numérica", data.select_dtypes(include=['int64','float64']).columns)

    # Histograma
    st.subheader(f"Distribución de {selected_col}")
    fig, ax = plt.subplots()
    sns.histplot(data[selected_col], kde=True, ax=ax)
    st.pyplot(fig)

    # Correlación
    st.subheader("Mapa de Correlación")
    corr = data.select_dtypes(include=['int64','float64']).corr()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    # Modelo simple (demo)
    if st.checkbox("Ejecutar modelo de ejemplo (Random Forest)"):
        if 'Price' in data.columns:
            data = data.dropna()
            X = data.drop('Price', axis=1).select_dtypes(include=['int64','float64'])
            y = data['Price']
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)
            preds = model.predict(X)
            mse = mean_squared_error(y, preds)
            st.write(f"Error cuadrático medio (MSE): {mse:,.2f}")
        else:
            st.warning("No se encontró la columna 'price' para entrenar el modelo.")
else:
    st.stop()
