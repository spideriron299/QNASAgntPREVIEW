
Copiar

import streamlit as st
import h5py
import pandas as pd
import pydeck as pdk
import numpy as np
import tempfile
import os
 
# Configuración de la página (DEBE ser la primera instrucción de Streamlit)
st.set_page_config(page_title="HDF5 Geo-Explorer", layout="wide")
 
st.title("🛰️ Visor de Archivos HDF5 con Mapa")
 
# --- Barra Lateral: Carga de Archivo ---
uploaded_file = st.sidebar.file_uploader("Sube tu archivo .h5 o .hdf5", type=['h5', 'hdf5'])
 
 
def collect_datasets(hdf_file):
    """Recorre recursivamente el archivo y devuelve todos los datasets."""
    found = []
 
    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            found.append(name)
 
    hdf_file.visititems(_visitor)
    return found
 
 
if uploaded_file is not None:
    # h5py necesita un archivo real en disco, no un objeto BytesIO
    # Guardamos el contenido en un archivo temporal y lo leemos desde ahí
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
 
    try:
        with h5py.File(tmp_path, 'r') as f:
            datasets = collect_datasets(f)
 
            if not datasets:
                st.sidebar.error("No se encontraron datasets en el archivo.")
            else:
                st.sidebar.success("Archivo cargado correctamente ✅")
 
                # --- Selección de Dataset ---
                target_dataset = st.sidebar.selectbox(
                    "Selecciona un Dataset para visualizar", datasets
                )
 
                data_node = f[target_dataset]
 
                if len(data_node.shape) < 2:
                    st.warning(
                        "El dataset seleccionado debe ser al menos bidimensional "
                        "para mostrarse como tabla/mapa."
                    )
                else:
                    # Leer datos (limitamos a 10 000 filas para rendimiento)
                    raw = data_node[:10_000]
 
                    # Si el dtype tiene campos nombrados (structured array) los usamos como columnas
                    if raw.dtype.names:
                        df = pd.DataFrame(raw)
                        # Decodificar bytes a str si es necesario
                        for col in df.columns:
                            if df[col].dtype == object:
                                try:
                                    df[col] = df[col].str.decode("utf-8")
                                except AttributeError:
                                    pass
                    else:
                        df = pd.DataFrame(raw)
 
                    st.subheader(f"Vista previa: `{target_dataset}`")
                    st.dataframe(df.head(100), use_container_width=True)
 
                    # --- Configuración del Mapa ---
                    st.divider()
                    st.subheader("🗺️ Configuración del Mapa")
 
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
 
                    if len(numeric_cols) < 2:
                        st.info(
                            "No hay suficientes columnas numéricas para trazar un mapa. "
                            "Se necesitan al menos una columna de latitud y una de longitud."
                        )
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            lat_col = st.selectbox(
                                "Columna de Latitud", numeric_cols, index=0
                            )
                        with col2:
                            lon_options = [c for c in numeric_cols if c != lat_col]
                            lon_col = st.selectbox(
                                "Columna de Longitud",
                                lon_options,
                                index=0 if lon_options else None,
                            )
 
                        if lon_col:
                            map_df = (
                                df[[lat_col, lon_col]]
                                .dropna()
                                .rename(columns={lat_col: "lat", lon_col: "lon"})
                            )
 
                            # Filtrar coordenadas fuera de rango válido
                            map_df = map_df[
                                map_df["lat"].between(-90, 90)
                                & map_df["lon"].between(-180, 180)
                            ]
 
                            if not map_df.empty:
                                st.pydeck_chart(
                                    pdk.Deck(
                                        map_style="mapbox://styles/mapbox/light-v9",
                                        initial_view_state=pdk.ViewState(
                                            latitude=map_df["lat"].mean(),
                                            longitude=map_df["lon"].mean(),
                                            zoom=3,
                                            pitch=0,
                                        ),
                                        layers=[
                                            pdk.Layer(
                                                "ScatterplotLayer",
                                                data=map_df,
                                                get_position="[lon, lat]",
                                                get_color="[200, 30, 0, 160]",
                                                get_radius=20_000,
                                                pickable=True,
                                            )
                                        ],
                                        tooltip={"text": "Lat: {lat}\nLon: {lon}"},
                                    )
                                )
                                st.caption(f"{len(map_df):,} puntos graficados.")
                            else:
                                st.error(
                                    "No se encontraron coordenadas válidas en las columnas seleccionadas. "
                                    "Verifica que los valores estén en rango (lat: −90…90, lon: −180…180)."
                                )
 
    finally:
        # Siempre eliminar el archivo temporal
        os.unlink(tmp_path)
 
else:
    st.info("Por favor, sube un archivo HDF5 desde la barra lateral para comenzar.")