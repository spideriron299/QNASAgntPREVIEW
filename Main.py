import streamlit as st
import h5py
import pandas as pd
import pydeck as pdk
import numpy as np
import tempfile
import os

# pyhdf es opcional: solo se importa si está instalado
try:
    from pyhdf.SD import SD, SDC
    PYHDF_AVAILABLE = True
except ImportError:
    PYHDF_AVAILABLE = False

st.set_page_config(page_title="HDF Geo-Explorer", layout="wide")
st.title("🛰️ Visor de Archivos HDF con Mapa")

# --- Barra Lateral: Carga de Archivo ---
uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo HDF4 o HDF5",
    type=['h5', 'hdf5', 'hdf', 'he4', 'he5', 'nc']
)


# ──────────────────────────────────────────────
# Helpers HDF5 (h5py)
# ──────────────────────────────────────────────
def collect_datasets_hdf5(hdf_file):
    found = []
    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            found.append(name)
    hdf_file.visititems(_visitor)
    return found


def read_hdf5(tmp_path):
    with h5py.File(tmp_path, 'r') as f:
        datasets = collect_datasets_hdf5(f)
        if not datasets:
            st.sidebar.error("No se encontraron datasets en el archivo.")
            return None

        st.sidebar.success("HDF5 cargado correctamente ✅")
        target_dataset = st.sidebar.selectbox("Selecciona un Dataset", datasets)
        data_node = f[target_dataset]

        if len(data_node.shape) < 2:
            st.warning("El dataset debe ser al menos bidimensional.")
            return None

        raw = data_node[:10_000]
        if raw.dtype.names:
            df = pd.DataFrame(raw)
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = df[col].str.decode("utf-8")
                    except AttributeError:
                        pass
        else:
            df = pd.DataFrame(raw)

    return df


# ──────────────────────────────────────────────
# Helpers HDF4 (pyhdf)
# ──────────────────────────────────────────────
def read_hdf4(tmp_path):
    if not PYHDF_AVAILABLE:
        st.error(
            "Para leer archivos HDF4 necesitas instalar **pyhdf**.\n\n"
            "Agrega `pyhdf` a tu `requirements.txt` y vuelve a desplegar."
        )
        return None

    sd = SD(tmp_path, SDC.READ)
    datasets = list(sd.datasets().keys())

    if not datasets:
        st.sidebar.error("No se encontraron datasets en el archivo HDF4.")
        return None

    st.sidebar.success("HDF4 cargado correctamente ✅")
    target_dataset = st.sidebar.selectbox("Selecciona un Dataset", datasets)
    sds = sd.select(target_dataset)
    raw = sds.get()
    sds.endaccess()
    sd.end()

    if raw.ndim < 2:
        st.warning("El dataset debe ser al menos bidimensional.")
        return None

    # Aplanar a 2D si tiene más dimensiones (tomar primer slice)
    if raw.ndim > 2:
        st.info(f"El dataset tiene forma {raw.shape}. Se muestra el primer slice `[0, ...]`.")
        raw = raw[0]

    return pd.DataFrame(raw[:10_000])


# ──────────────────────────────────────────────
# Renderizado del mapa (común a ambos formatos)
# ──────────────────────────────────────────────
def render_map_section(df):
    st.subheader("Vista previa del Dataset")
    st.dataframe(df.head(100), use_container_width=True)

    st.divider()
    st.subheader("🗺️ Configuración del Mapa")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.info("No hay suficientes columnas numéricas para trazar un mapa.")
        return

    col1, col2 = st.columns(2)
    with col1:
        lat_col = st.selectbox("Columna de Latitud", numeric_cols, index=0)
    with col2:
        lon_options = [c for c in numeric_cols if c != lat_col]
        lon_col = st.selectbox("Columna de Longitud", lon_options, index=0 if lon_options else None)

    if not lon_col:
        return

    map_df = (
        df[[lat_col, lon_col]]
        .dropna()
        .rename(columns={lat_col: "lat", lon_col: "lon"})
    )
    map_df = map_df[map_df["lat"].between(-90, 90) & map_df["lon"].between(-180, 180)]

    if map_df.empty:
        st.error("No se encontraron coordenadas válidas (lat: −90…90, lon: −180…180).")
        return

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


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if uploaded_file is not None:
    ext = uploaded_file.name.rsplit('.', 1)[-1].lower()
    is_hdf4 = ext in ('hdf', 'he4')

    suffix = f".{ext}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if is_hdf4:
            df = read_hdf4(tmp_path)
        else:
            df = read_hdf5(tmp_path)

        if df is not None:
            render_map_section(df)

    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
    finally:
        os.unlink(tmp_path)

else:
    st.info("Por favor, sube un archivo HDF4 (.hdf, .he4) o HDF5 (.h5, .hdf5) desde la barra lateral.")
