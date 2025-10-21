import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
import seaborn as sns

# Configuración general 
st.set_page_config(page_title="Check-in / Check-out — Di Donato", layout="wide")

COLOR_PRINCIPAL = "#009C7D"
COLOR_CLARO = "#E6F4F1"
COLOR_ROJO = "#D23F2A"
COLOR_TEXTO = "#333"

# Carga de archivo
st.title("Reporte Check-in / Check-out")

archivo = st.file_uploader("📂 Subí el archivo Excel bajado de AXUM", type=["xlsx"])
if archivo is None:
    st.info("⬆ Subí el archivo para comenzar.")
    st.stop()

# Lectura y limpieza
df = pd.read_excel(archivo)
df.columns = df.columns.str.strip()

required = ["Vendedor", "Es Valido", "Fecha Checkin", "Fecha Checkout", "Tiempo en PDV", "Cliente"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Faltan columnas esperadas en el archivo: {missing}")
    st.stop()

df["Fecha Checkin"] = pd.to_datetime(df["Fecha Checkin"], dayfirst=True, errors="coerce")
df["Fecha Checkout"] = pd.to_datetime(df["Fecha Checkout"], dayfirst=True, errors="coerce")
df["Tiempo_PDVTd"] = pd.to_timedelta(df["Tiempo en PDV"], errors="coerce")

df["Total_Seconds"] = df["Tiempo_PDVTd"].dt.total_seconds()
df["Total_Minutes"] = df["Total_Seconds"] / 60.0
df["Total_Seconds"] = df["Total_Seconds"].fillna(0)
df["Total_Minutes"] = df["Total_Minutes"].fillna(0)

def secs_to_sexagesimal_float(seconds):
    if pd.isna(seconds):
        return np.nan
    total_seconds = int(round(seconds))
    horas = total_seconds // 3600
    minutos = (total_seconds % 3600) // 60
    return float(f"{horas}.{minutos:02d}")

def secs_to_HHMM_string(seconds):
    if pd.isna(seconds):
        return None
    total_seconds = int(round(seconds))
    horas = total_seconds // 3600
    minutos = (total_seconds % 3600) // 60
    return f"{horas}:{minutos:02d}"

df["Tiempo_Sexagesimal"] = df["Total_Seconds"].apply(secs_to_sexagesimal_float)
df["Tiempo_HHMM"] = df["Total_Seconds"].apply(secs_to_HHMM_string)
df["Es Valido"] = df["Es Valido"].astype(str).str.strip().str.upper()

# Filtros
st.sidebar.header("Filtros")
filtro_validez = st.sidebar.selectbox("Filtrar por validez:", ["Solo SI", "Todos", "Solo NO"])

if filtro_validez == "Solo SI":
    df_f = df[df["Es Valido"] == "SI"].copy()
elif filtro_validez == "Solo NO":
    df_f = df[df["Es Valido"] == "NO"].copy()
else:
    df_f = df.copy()

min_fecha = df["Fecha Checkin"].min()
max_fecha = df["Fecha Checkin"].max()

if pd.isna(min_fecha) or pd.isna(max_fecha):
    st.error("No hay fechas válidas en 'Fecha Checkin'.")
    st.stop()

rango = st.sidebar.date_input(
    "Rango de fechas (Fecha Check-in)",
    value=(min_fecha.date(), max_fecha.date())
)
start_date = pd.to_datetime(rango[0])
end_date = pd.to_datetime(rango[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
df_f = df_f[(df_f["Fecha Checkin"] >= start_date) & (df_f["Fecha Checkin"] <= end_date)]

# Función heatmap
def heatmap_style(df_style, cols):
    return (
        df_style.style.background_gradient(
            cmap=sns.light_palette(COLOR_PRINCIPAL, as_cmap=True), subset=cols
        )
        .set_properties(**{"text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [("font-weight", "bold"), ("text-align", "center"), ("background-color", COLOR_CLARO)]}
        ])
    )

# Tabs 
tabs = st.tabs(["Horas vendedores", "Datos Generales", "Clientes", "Valores Extremos"])

# Principal 
with tabs[0]:
    st.subheader("Información por Vendedor")
    st.markdown("> Ordenado por defecto según las horas totales. Podes cambiarlo tocando los encabezados")

    agg = df_f.groupby("Vendedor", as_index=False).agg(
        Registros=("Cliente", "count"),
        Sum_Seconds=("Total_Seconds", "sum"),
        Mean_Seconds=("Total_Seconds", "mean")
    )

    agg["Horas"] = agg["Sum_Seconds"].apply(secs_to_sexagesimal_float)
    agg["Promedio (h)"] = agg["Mean_Seconds"].apply(secs_to_sexagesimal_float)

    # Limitar a 2 decimales y quitar ceros innecesarios
    agg["Horas"] = agg["Horas"].map(lambda x: f"{x:.2f}".rstrip("0").rstrip("."))
    agg["Promedio (h)"] = agg["Promedio (h)"].map(lambda x: f"{x:.2f}".rstrip("0").rstrip("."))

    agg = agg.sort_values("Sum_Seconds", ascending=False)
    show_cols = ["Vendedor", "Registros", "Horas", "Promedio (h)"]
    display_df = agg[show_cols].copy()
    st.dataframe(heatmap_style(display_df, ["Registros", "Horas", "Promedio (h)"]), use_container_width=True)

# Datos Generales
with tabs[1]:
    st.subheader("Resumen General")

    mean_seconds_overall = df_f["Total_Seconds"].mean()
    mean_sexagesimal = secs_to_sexagesimal_float(mean_seconds_overall)
    total_vendedores = df_f["Vendedor"].nunique()
    total_registros = len(df_f)

    c1, c2, c3 = st.columns(3)
    c1.metric("Promedio general (h)", f"{mean_sexagesimal}")
    c2.metric("Cantidad de vendedores", total_vendedores)
    c3.metric("Cantidad de registros", total_registros)

    # --- Total de horas por día ---
    df_day = df_f.groupby(df_f["Fecha Checkin"].dt.date)["Total_Seconds"].sum().reset_index()
    df_day["Horas"] = df_day["Total_Seconds"].apply(secs_to_sexagesimal_float)
    fig_min = px.bar(
        df_day, x="Fecha Checkin", y="Horas", title="Horas totales por día",
        color_discrete_sequence=[COLOR_PRINCIPAL], text_auto=True
    )
    fig_min.update_layout(
        yaxis_title="Horas",
        xaxis_tickangle=-45,
        bargap=0.2
    )
    st.plotly_chart(fig_min, use_container_width=True)

    # Cantidad de registros por día
    df_count = df_f.groupby(df_f["Fecha Checkin"].dt.date)["Cliente"].count().reset_index()
    df_count.rename(columns={"Cliente": "Registros"}, inplace=True)
    fig_reg = px.bar(
        df_count, x="Fecha Checkin", y="Registros", title="Cantidad de registros por día",
        color_discrete_sequence=[COLOR_PRINCIPAL], text_auto=True
    )
    fig_reg.update_layout(
        xaxis_tickangle=-45,
        bargap=0.2
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    # Distribución horaria 
    df_f["Hora_Checkin"] = df_f["Fecha Checkin"].dt.hour
    df_hour = df_f.groupby("Hora_Checkin")["Cliente"].count().reset_index()
    df_hour.rename(columns={"Cliente": "Cantidad_Checkins"}, inplace=True)

    fig_hour = px.bar(
        df_hour, x="Hora_Checkin", y="Cantidad_Checkins",
        title="Distribución horaria de check-ins",
        color_discrete_sequence=[COLOR_PRINCIPAL], text_auto=True
    )
    fig_hour.update_layout(
        xaxis_title="Hora del día",
        yaxis_title="Cantidad de check-ins",
        bargap=0.15
    )
    st.plotly_chart(fig_hour, use_container_width=True)

# Clientes
with tabs[2]:
    st.subheader("Resumen por Cliente")
    agg_c = df_f.groupby("Cliente", as_index=False).agg(
        Registros=("Vendedor", "count"),
        Sum_Seconds=("Total_Seconds", "sum"),
        Mean_Seconds=("Total_Seconds", "mean")
    )
    agg_c["Horas"] = agg_c["Sum_Seconds"].apply(secs_to_sexagesimal_float)
    agg_c["Promedio (h)"] = agg_c["Mean_Seconds"].apply(secs_to_sexagesimal_float)
    agg_c["Horas"] = agg_c["Horas"].map(lambda x: f"{x:.2f}".rstrip("0").rstrip("."))
    agg_c["Promedio (h)"] = agg_c["Promedio (h)"].map(lambda x: f"{x:.2f}".rstrip("0").rstrip("."))
    agg_c = agg_c.sort_values("Sum_Seconds", ascending=False)
    show_cols_c = ["Cliente", "Registros", "Horas", "Promedio (h)"]
    display_clients = agg_c[show_cols_c].copy()
    st.dataframe(heatmap_style(display_clients, ["Registros", "Horas", "Promedio (h)"]), use_container_width=True)

# Valores Extremos
with tabs[3]:
    st.subheader("🚨 Valores Extremos")
    st.markdown(
        "Este módulo identifica registros atípicos en 'Tiempo en PDV'"
        "La **sensibilidad** define cuán estricta es la detección: "
        "si aumentás la sensibilidad, el modelo marcará más registros como posibles valores extremos."
    )

    sensibilidad = st.slider(
        "Sensibilidad del análisis (0.01 a 0.1):",
        min_value=0.01, max_value=0.1, value=0.01, step=0.01,
        help="A mayor sensibilidad, más registros se consideran valores extremos."
    )

    # Por registro individual 
    clf_ind = IsolationForest(contamination=sensibilidad, random_state=42)
    df_anom = df_f[["Cliente", "Total_Seconds"]].copy()
    df_anom["Total_Minutes"] = df_anom["Total_Seconds"] / 60
    df_anom["Predicción"] = clf_ind.fit_predict(df_anom[["Total_Minutes"]])
    df_anom["Estado"] = np.where(df_anom["Predicción"] == -1, "Valor extremo", "Normal")
    df_anom["Horas"] = df_anom["Total_Seconds"].apply(secs_to_sexagesimal_float)

    fig1 = px.scatter(
        df_anom.reset_index(),
        x="index", y="Horas", color="Estado", hover_data=["Cliente"],
        color_discrete_map={"Normal": COLOR_PRINCIPAL, "Valor extremo": COLOR_ROJO},
        title="Tiempos por registro individual"
    )
    fig1.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0.5, color="white")))
    fig1.update_layout(
        yaxis_title="Horas",
        xaxis_title="Registro",
        yaxis_range=[0, df_anom["Horas"].max()*1.1],
        margin=dict(t=60, b=50)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Por promedio de tiempo por vendedor 
    df_mean = df_f.groupby("Vendedor", as_index=False).agg(Mean_Seconds=("Total_Seconds", "mean"))
    clf_vend = IsolationForest(contamination=sensibilidad, random_state=42)
    df_mean["Predicción"] = clf_vend.fit_predict((df_mean["Mean_Seconds"] / 60).to_frame())
    df_mean["Estado"] = np.where(df_mean["Predicción"] == -1, "Valor extremo", "Normal")
    df_mean["Promedio"] = df_mean["Mean_Seconds"].apply(secs_to_sexagesimal_float)

    fig2 = px.scatter(
        df_mean, x="Vendedor", y="Promedio", color="Estado",
        color_discrete_map={"Normal": COLOR_PRINCIPAL, "Valor extremo": COLOR_ROJO},
        title="Promedio por vendedor"
    )
    fig2.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="white")))
    fig2.update_layout(
        yaxis_title="Horas",
        xaxis_title="Vendedor",
        margin=dict(t=60, b=50)
    )
    st.plotly_chart(fig2, use_container_width=True)
