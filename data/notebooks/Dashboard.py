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

header = html.Div([
    html.H2("Planificación de Fondos", className="app-title"),
    html.Div(className="app-divider")
], className="pt-4")  

app.layout = dbc.Container([
    header
], fluid=True)


# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)