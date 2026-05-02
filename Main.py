import streamlit as st
import h5py
import pandas as pd
import pydeck as pdk
import numpy as np
import tempfile
import os

# netCDF4 para HDF5/NetCDF
try:
    import netCDF4 as nc
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

# GDAL para HDF4
try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

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
    return df, None  # (df, geo_df)


# ──────────────────────────────────────────────
# NetCDF con netCDF4
# ──────────────────────────────────────────────
def read_netcdf(tmp_path):
    if not NETCDF4_AVAILABLE:
        st.error("netCDF4 no está disponible.")
        return None, None
    try:
        ds = nc.Dataset(tmp_path, 'r')
    except Exception as e:
        st.error(f"No se pudo abrir el archivo NetCDF: {e}")
        return None, None

    var_names = [v for v in ds.variables if ds.variables[v].ndim >= 2]
    if not var_names:
        st.sidebar.error("No se encontraron variables 2D.")
        ds.close()
        return None, None

    st.sidebar.success("NetCDF cargado correctamente ✅")
    target = st.sidebar.selectbox("Selecciona una Variable", var_names)
    raw = np.array(ds.variables[target][:])
    ds.close()

    if raw.ndim > 2:
        st.info(f"Variable con forma {raw.shape}. Se muestra el primer slice `[0, ...]`.")
        raw = raw[0]

    return pd.DataFrame(raw[:10_000]), None


# ──────────────────────────────────────────────
# HDF4 con GDAL — calcula lat/lon automáticamente
# ──────────────────────────────────────────────
def raster_to_geo_df(band_ds, max_points=50_000):
    """
    Convierte un raster GDAL a un DataFrame con columnas
    lat, lon, value. Reproyecta a WGS84 si es necesario.
    """
    gt = band_ds.GetGeoTransform()   # (x_origin, px_w, rot, y_origin, rot, px_h)
    src_srs = band_ds.GetSpatialRef()
    cols = band_ds.RasterXSize
    rows = band_ds.RasterYSize

    # Crear SRS destino WGS84
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    need_reproject = src_srs is not None and not src_srs.IsSame(wgs84)
    if need_reproject:
        transform = osr.CoordinateTransformation(src_srs, wgs84)

    # Submuestrear si hay demasiados píxeles
    total = rows * cols
    if total > max_points:
        step = int(np.ceil(np.sqrt(total / max_points)))
    else:
        step = 1

    band = band_ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    data = band.ReadAsArray()

    records = []
    for r in range(0, rows, step):
        for c in range(0, cols, step):
            val = float(data[r, c])
            if nodata is not None and val == nodata:
                continue
            # Pixel center → projected coordinates
            x = gt[0] + (c + 0.5) * gt[1] + (r + 0.5) * gt[2]
            y = gt[3] + (c + 0.5) * gt[4] + (r + 0.5) * gt[5]

            if need_reproject:
                try:
                    lon, lat, _ = transform.TransformPoint(x, y)
                except Exception:
                    continue
            else:
                lon, lat = x, y

            if -90 <= lat <= 90 and -180 <= lon <= 180:
                records.append({'lat': lat, 'lon': lon, 'value': val})

    return pd.DataFrame(records)


def read_hdf4(tmp_path):
    if not GDAL_AVAILABLE:
        st.error("GDAL no está disponible.")
        return None, None

    ds = gdal.Open(tmp_path)
    if ds is None:
        st.error("GDAL no pudo abrir el archivo.")
        return None, None

    subdatasets = ds.GetSubDatasets()
    ds = None

    if not subdatasets:
        st.sidebar.error("No se encontraron subdatasets HDF4.")
        return None, None

    st.sidebar.success("HDF4 cargado correctamente ✅")
    options = [s[1] for s in subdatasets]
    paths   = [s[0] for s in subdatasets]

    target_label = st.sidebar.selectbox("Selecciona un Dataset", options)
    target_path  = paths[options.index(target_label)]

    band_ds = gdal.Open(target_path)
    if band_ds is None:
        st.error("No se pudo abrir el subdataset.")
        return None, None

    # DataFrame tabular (vista previa)
    raw = band_ds.GetRasterBand(1).ReadAsArray()
    df_preview = pd.DataFrame(raw[:100])  # solo primeras 100 filas para la tabla

    # DataFrame georreferenciado para el mapa
    with st.spinner("Calculando coordenadas lat/lon..."):
        geo_df = raster_to_geo_df(band_ds)

    band_ds = None
    return df_preview, geo_df


# ──────────────────────────────────────────────
# Mapa — modo automático (geo_df) o manual (df)
# ──────────────────────────────────────────────
def render_map_section(df, geo_df=None):
    st.subheader("Vista previa del Dataset")
    st.dataframe(df.head(100), use_container_width=True)
    st.divider()
    st.subheader("🗺️ Mapa")

    # ── Modo automático: geo_df ya tiene lat/lon/value ──
    if geo_df is not None and not geo_df.empty:
        map_df = geo_df[['lat', 'lon']].copy()

        # Normalizar value para colorear puntos (rojo = alto, azul = bajo)
        if 'value' in geo_df.columns:
            vmin = geo_df['value'].quantile(0.02)
            vmax = geo_df['value'].quantile(0.98)
            norm = ((geo_df['value'] - vmin) / (vmax - vmin + 1e-9)).clip(0, 1)
            map_df['r'] = (norm * 200).astype(int)
            map_df['g'] = 30
            map_df['b'] = ((1 - norm) * 200).astype(int)
        else:
            map_df['r'], map_df['g'], map_df['b'] = 200, 30, 0

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=map_df['lat'].mean(),
                longitude=map_df['lon'].mean(),
                zoom=4, pitch=0,
            ),
            layers=[pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_color="[r, g, b, 180]",
                get_radius=5_000,
                pickable=True,
            )],
            tooltip={"text": "Lat: {lat}\nLon: {lon}"},
        ))
        st.caption(f"{len(map_df):,} puntos graficados (coordenadas calculadas automáticamente desde la geotransformación del raster).")
        return

    # ── Modo manual: el usuario elige columnas ──
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
        df[[lat_col, lon_col]].dropna()
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

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if ext in ('h5', 'hdf5'):
            df, geo_df = read_hdf5(tmp_path)
        elif ext in ('nc', 'nc4', 'he5'):
            df, geo_df = read_netcdf(tmp_path)
        else:  # hdf, he4 → HDF4
            df, geo_df = read_hdf4(tmp_path)

        if df is not None:
            render_map_section(df, geo_df)

    except Exception as e:
        st.error(f"Error inesperado: {e}")
    finally:
        os.unlink(tmp_path)

else:
    st.info("Sube un archivo HDF4 (.hdf, .he4), HDF5 (.h5, .hdf5) o NetCDF (.nc) desde la barra lateral.")