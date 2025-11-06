import datetime as dt
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json, joblib


# App setup
# -----------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP,
                          "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css"],
    title="Dashboard Provisionamiento",
)

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
    kpi_cards,
    # agregarlo como un dbc row, con una sola columna 
    # className hace referencia a una clase de CSS que tiene estilos visuales predefinidos
    dbc.Row([dbc.Col(graph_card, md=12)], className="mt-3"),
    # tambien agregar la tabla como un row con una sola columna
    dbc.Row([dbc.Col(table_card, md=12)], className="mt-3 mb-5"),
], fluid=True)


# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)