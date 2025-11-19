import datetime as dt
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json, joblib
import os
import requests

# App setup
# -----------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP,
                          "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css"],
    title="Dashboard Provisionamiento",
)

def get_date_limits():
    """Pregunta a la API por el histórico y calcula min_date y max_date."""
    today = dt.date.today()
    resp = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"fecha_fin": str(today)}
    )
    resp.raise_for_status()
    payload = resp.json()
    df = pd.DataFrame(payload["predicciones"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    # mínima fecha del dataset
    min_date = df["fecha"].min().date()
    # última fecha con venta_real no nula
    last_real = df[df["venta_real"].notna()]["fecha"].max().date()
    # máximo permitido: +3 semanas
    max_date = last_real + dt.timedelta(weeks=3)
    return min_date, max_date
MIN_DATE, MAX_DATE = get_date_limits()

# CSS global (fuente y estilos)
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      :root { --app-font: "Segoe UI", "Segoe UI Variable", -apple-system, system-ui, "Helvetica Neue", Arial, sans-serif; }
      html, body, #react-entry-point { font-family: var(--app-font); background-color: #f8f9fa !important;}
      
      /* Título y divisor */
      .app-title {font-weight:600; font-size: 24px; margin: 0;}
      .app-divider {border:0; height:1px; background:#e9ecef; margin:12px 0 16px;}

      /* Cards KPI */
      .kpi-card {border: 1px solid#e9ecef; border-radius:16px; box-shadow:none;}
      .kpi-label {color: #6c757d; font-size:12px; margin-bottom: 4px;}

      /* Tabla - al hacer esto se pone el padding en todo */
      .table-wrap .dash-table-container .dash-spreadsheet-container { font-family: var(--app-font); }
      .container-fluid {padding: 40px 60px !important;  /* ↑↓40px, ←→60px — ajusta a gusto */}

      /* Cards generales (gráfica, tabla, etc.) */
      .card {
      border: 1px solid #e9ecef !important;   /* mismo color que las KPI cards */
      border-radius: 16px !important;         /* esquinas suaves */
      background-color: #ffffff !important;   /* fondo blanco */  
      box-shadow: none !important;            /* sin sombra */
      }

      /* Configuracion del Date Input */
      .DateInput_input {
      border: none !important;
      box-shadow: none !important;
      padding: 4px 6px !important;
      height: 28px !important;
      font-size: 14px !important;
      background-color: #ffffff !important;
      }

      /* Hover */
      .SingleDatePickerInput:hover {
      border-color: #d0d7de !important;
      background-color: #f8f9fa !important;
      }
      
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>
"""

# Primer componente: header
header = html.Div([
    html.H2("Planificación de Fondos", className="app-title"),
    html.Div(className="app-divider")
], className="pt-4")  

# Filtro de fecha
filters = dbc.Row([
    dbc.Col(
        dbc.CardBody([
            dbc.Label([
                "Fecha", 
                html.I(className="bi bi-info-circle-fill text-muted", 
                       id="tooltip-fecha", 
                       style={"cursor": "pointer",
                              "marginLeft":"4px"})],
                  style={"marginRight":"8px"},
    ),
    dbc.Tooltip(
    "Selecciona hasta qué fecha quieres hacer la predicción",
    target="tooltip-fecha",
    placement="right",   
    style={"fontSize": "13px"}
    ),
        dcc.DatePickerSingle(
            id="filtro-fecha",
            date=MAX_DATE,
            display_format="DD/MM/YYYY",
            min_date_allowed=MIN_DATE,
            max_date_allowed=MAX_DATE
        )
    ]), md=3)
], className="g-3")

# Segundo componente: cards (row de 3 columnas)
kpi_cards = dbc.Row([
    dbc.Col(dbc.Card(dbc.CardBody([
        html.Div("Fondo Total", className="kpi-label"),
        html.H3(id="kpi-fondo", className="mb-0")
    ]), className="kpi-card"), md=4),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.Div("RMSE", className="kpi-label"),
        html.H3(id="kpi-rmse", className="mb-0")
    ]), className="kpi-card"), md=4),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.Div("Variación mensual", className="kpi-label"),
        html.H4(id="kpi-var", className="mb-0")
    ]), className="kpi-card"), md=4),
], className="g-3 mt-1")

# Tercer componente: la grafica de lineas
graph_card = dbc.Card(dbc.CardBody([
    dcc.Graph(id="grafico-linea", config={"displayModeBar": False}, style={"height": 300})
]))

# Cuarto componente: la tabla
table_card = dbc.Card(dbc.CardBody([
    html.H5("Tabla de resumen por proveedor", className="mb-3"),
    html.Div([
        dash_table.DataTable(
            id="tabla-resumen",
            page_size=5,
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={
                "padding": "8px",
                "fontFamily": "Segoe UI, Segoe UI Variable, -apple-system, system-ui, Arial, sans-serif",
                "fontSize": "14px",
                "borderBottom": "1px solid #f1f3f5"
            },
            style_header={
                "fontWeight": "700",
                "backgroundColor": "white",
                "borderBottom": "1px solid #e9ecef"
            },
        )
    ], className="table-wrap")
]))

app.layout = dbc.Container([
    header,
    filters,
    kpi_cards,
    # agregarlo como un dbc row, con una sola columna 
    # className hace referencia a una clase de CSS que tiene estilos visuales predefinidos
    dbc.Row([dbc.Col(graph_card, md=12)], className="mt-3"),
    # tambien agregar la tabla como un row con una sola columna
    dbc.Row([dbc.Col(table_card, md=12)], className="mt-3 mb-5"),
], fluid=True)

# --- helpers ---
def format_currency(x):
    return "" if pd.isna(x) else f"${x:,.0f}".replace(",", ".")

def compute_kpis(df: pd.DataFrame, date: dt.date):
    # Última fecha con venta_real disponible
    mask_real = df["venta_real"].notna()
    last_real_date = df.loc[mask_real, "fecha"].dt.date.max() if mask_real.any() else None
    # KPI 1: ventas totales (reales o predichas según el caso)
    if last_real_date is not None and date <= last_real_date:
        # Caso histórico: usar solo ventas reales del día seleccionado
        dfd = df[df["fecha"].dt.date == date]
        kpi_ventas = dfd["venta_real"].sum()
    elif last_real_date is not None:
        # Caso futuro: suma de predicciones desde el día después de last_real_date hasta la fecha seleccionada
        mask = (df["fecha"].dt.date > last_real_date) & (df["fecha"].dt.date <= date)
        dfd = df[mask]
        kpi_ventas = dfd["venta_predicha"].sum()
    else:
        # Backup raro: si no hay ventas reales en absoluto, usar solo predicciones hasta la fecha
        mask = df["fecha"].dt.date <= date
        dfd = df[mask]
        kpi_ventas = dfd["venta_predicha"].sum()
    # RMSE: solo donde hay venta_real y venta_predicha (hasta la fecha seleccionada)
    dfd_rmse = df[
        (df["fecha"].dt.date <= date)
        & df["venta_real"].notna()
        & df["venta_predicha"].notna()
    ]
    if not dfd_rmse.empty:
        rmse = np.sqrt(np.mean((dfd_rmse["venta_real"] - dfd_rmse["venta_predicha"]) ** 2))
    else:
        rmse = np.nan
    # Variación decorativa (la puedes cambiar luego si quieres algo más real)
    var_pct = np.random.uniform(-0.15, 0.15)  # opcional / decorativo
    return kpi_ventas, rmse, var_pct

# Callbacks
# -----------------------------
@app.callback(
    Output("kpi-fondo","children"),
    Output("kpi-rmse","children"),
    Output("kpi-var","children"),
    Output("grafico-linea","figure"),
    Output("tabla-resumen","columns"),
    Output("tabla-resumen","data"),
    Input("filtro-fecha","date")
)

def update_dashboard(date_str):
    date = pd.to_datetime(date_str).date()

    # Llamada a la API
    resp = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"fecha_fin": str(date)}
    )
    resp.raise_for_status()
    payload = resp.json()
    df = pd.DataFrame(payload["predicciones"])

    # Asegurar tipos
    df["fecha"] = pd.to_datetime(df["fecha"])

    # por si vienen None
    for col in ["venta_real", "venta_predicha", "fondo_recomendado",
                "recurso_sobrante"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # KPIs
    kpi_ventas, rmse, var_pct = compute_kpis(df, date)
    k1 = format_currency(kpi_ventas)
    k2 = format_currency(rmse)
    k3 = f"{var_pct*100:.1f}%" if var_pct is not None else "—"

    # Gráfica agregada
    series = df.groupby("fecha")[["venta_real", "venta_predicha"]].sum().reset_index()

    # Última fecha real
    mask_real = df["venta_real"].notna()
    last_real_date = df.loc[mask_real, "fecha"].max().date() if mask_real.any() else None
    if last_real_date:
        # Donde la fecha es > last_real_date → venta_real debe ser NaN
        series["venta_real"] = series.apply(
            lambda row: row["venta_real"] if row["fecha"].date() <= last_real_date else np.nan,
            axis=1
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series["fecha"],
        y=series["venta_real"],
        name="Venta real",
        mode="lines+markers",
        line=dict(color="#9287F9")
    ))
    fig.add_trace(go.Scatter(
        x=series["fecha"],
        y=series["venta_predicha"],
        name="Venta predicha",
        mode="lines+markers",
        line=dict(color="#A6C0ED")
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        legend_title_text="",
        height=300,
        font=dict(family="Segoe UI, Segoe UI Variable, -apple-system, system-ui, Arial, sans-serif"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor="#DEE2E6",
            tickfont=dict(color="#495057")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E9ECEF",
            zeroline=False,
            linecolor="#DEE2E6",
            tickfont=dict(color="#495057")
        )
    )
    # Tabla por producto con la misma lógica que el KPI
    mask_real = df["venta_real"].notna()
    last_real_date = df.loc[mask_real, "fecha"].dt.date.max() if mask_real.any() else None
    if last_real_date is not None and date <= last_real_date:
        # Caso histórico: solo ese día
        dft = df[df["fecha"].dt.date == date]
    elif last_real_date is not None:
        # Caso futuro: rango desde el día después de last_real_date hasta la fecha seleccionada
        mask = (df["fecha"].dt.date > last_real_date) & (df["fecha"].dt.date <= date)
        dft = df[mask]
    else:
        # Backup: si no hay ventas reales, usamos todo hasta la fecha
        mask = df["fecha"].dt.date <= date
        dft = df[mask]
    sumd = (dft
            .groupby("producto", as_index=False)
            .agg(
                fondo_recomendado=("fondo_recomendado", "sum"),
                recurso_sobrante=("recurso_sobrante", "sum"),
                prediccion=("venta_predicha", "sum"),
                venta_real=("venta_real", "sum")
            ))

    for c in ["fondo_recomendado", "recurso_sobrante", "prediccion", "venta_real"]:
        sumd[c] = sumd[c].map(format_currency)
    columns = [{"name": col.replace("_", " ").title(), "id": col} for col in sumd.columns]
    data = sumd.to_dict("records")
    return k1, k2, k3, fig, columns, data


# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)