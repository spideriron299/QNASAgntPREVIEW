import streamlit as st
import h5py
import pandas as pd
import pydeck as pdk
import numpy as np

# Configuración de la página (DEBE ser la primera instrucción de Streamlit)
st.set_page_config(page_title="HDF5 Geo-Explorer", layout="wide")

st.title("🛰️ Visor de Archivos HDF5 con Mapa")

# --- Barra Lateral: Carga de Archivo ---
uploaded_file = st.sidebar.file_uploader("Sube tu archivo .h5 o .hdf5", type=['h5', 'hdf5'])

def get_datasets(name, obj):
    """Función recursiva para listar todos los datasets en el archivo"""
    if isinstance(obj, h5py.Dataset):
        datasets.append(name)

if uploaded_file is not None:
    # Guardar temporalmente para que h5py lo lea (h5py no lee directamente BytesIO con facilidad)
    with h5py.File(uploaded_file, 'r') as f:
        datasets = []
        f.visititems(get_datasets)
        
        st.sidebar.success("Archivo cargado correctamente")
        
        # --- Selección de Dataset ---
        target_dataset = st.sidebar.selectbox("Selecciona un Dataset para visualizar", datasets)
        
        # Extraer datos
        data_node = f[target_dataset]
        
        if len(data_node.shape) < 2:
            st.warning("El dataset seleccionado debe ser al menos bidimensional para mostrarse como tabla/mapa.")
        else:
            # Convertir a DataFrame (limitando filas para rendimiento)
            data_array = data_node[:]
            df = pd.DataFrame(data_array)
            
            st.subheader(f"Vista previa: `{target_dataset}`")
            st.write(df.head(100))
            
            # --- Configuración del Mapa ---
            st.divider()
            st.subheader("Configuración del Mapa")
            
            col1, col2 = st.columns(2)
            with col1:
                lat_col = st.selectbox("Columna de Latitud", df.columns, index=0)
            with col2:
                lon_col = st.selectbox("Columna de Longitud", df.columns, index=1)
                
            # Limpiar datos para el mapa (quitar NaNs)
            map_df = df[[lat_col, lon_col]].dropna()
            map_df.columns = ['lat', 'lon']

            # Renderizar Mapa con Pydeck
            if not map_df.empty:
                st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude=map_df['lat'].mean(),
                        longitude=map_df['lon'].mean(),
                        zoom=3,
                        pitch=0,
                    ),
                    layers=[
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=map_df,
                            get_position='[lon, lat]',
                            get_color='[200, 30, 0, 160]',
                            get_radius=20000, # Ajusta según la escala de tus datos
                        ),
                    ],
                ))
            else:
                st.error("No se encontraron coordenadas válidas en las columnas seleccionadas.")

else:
    st.info("Por favor, sube un archivo HDF5 desde la barra lateral para comenzar.")