import streamlit as st
import h5py
import pandas as pd
import pydeck as pdk
import numpy as np
import tempfile
import os

# netCDF4 puede leer HDF4 y HDF5, además de NetCDF
try:
    import netCDF4 as nc
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

st.set_page_config(page_title="HDF Geo-Explorer", layout="wide")
st.title("🛰️ Visor de Archivos HDF con Mapa")

uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo HDF4, HDF5 o NetCDF",
    type=['h5', 'hdf5', 'hdf', 'he4', 'he5', 'nc', 'nc4']
)


# ──────────────────────────────────────────────
# HDF5 con h5py
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
        target = st.sidebar.selectbox("Selecciona un Dataset", datasets)
        node = f[target]

        if node.ndim < 2:
            st.warning("El dataset debe ser al menos bidimensional.")
            return None

        raw = node[:10_000]
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
# HDF4 / NetCDF con netCDF4
# ──────────────────────────────────────────────
def read_hdf4_nc(tmp_path):
    if not NETCDF4_AVAILABLE:
        st.error("Instala **netCDF4** en tu `requirements.txt` para leer este formato.")
        return None

    try:
        ds = nc.Dataset(tmp_path, 'r')
    except Exception as e:
        st.error(f"No se pudo abrir el archivo: {e}")
        return None

    var_names = list(ds.variables.keys())
    if not var_names:
        st.sidebar.error("No se encontraron variables en el archivo.")
        ds.close()
        return None

    st.sidebar.success("HDF4/NetCDF cargado correctamente ✅")
    target = st.sidebar.selectbox("Selecciona una Variable", var_names)
    var = ds.variables[target]

    if var.ndim < 2:
        st.warning("La variable debe ser al menos bidimensional.")
        ds.close()
        return None

    raw = np.array(var[:])
    ds.close()

    # Aplanar a 2D tomando el primer slice si tiene más dimensiones
    if raw.ndim > 2:
        st.info(f"Variable con forma {raw.shape}. Se muestra el primer slice `[0, ...]`.")
        raw = raw[0]

    return pd.DataFrame(raw[:10_000])


# ──────────────────────────────────────────────
# Mapa
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

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=map_df["lat"].mean(),
            longitude=map_df["lon"].mean(),
            zoom=3, pitch=0,
        ),
        layers=[pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_color="[200, 30, 0, 160]",
            get_radius=20_000,
            pickable=True,
        )],
        tooltip={"text": "Lat: {lat}\nLon: {lon}"},
    ))
    st.caption(f"{len(map_df):,} puntos graficados.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if uploaded_file is not None:
    ext = uploaded_file.name.rsplit('.', 1)[-1].lower()
    is_hdf5 = ext in ('h5', 'hdf5', 'he5')

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if is_hdf5:
            df = read_hdf5(tmp_path)
        else:
            # HDF4 (.hdf, .he4) y NetCDF (.nc, .nc4) via netCDF4
            df = read_hdf4_nc(tmp_path)

        if df is not None:
            render_map_section(df)

    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
    finally:
        os.unlink(tmp_path)

else:
    st.info("Sube un archivo HDF4 (.hdf, .he4), HDF5 (.h5, .hdf5) o NetCDF (.nc) desde la barra lateral.")