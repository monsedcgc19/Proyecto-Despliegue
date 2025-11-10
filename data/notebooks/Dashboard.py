import datetime as dt
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json, joblib
import os


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
            date=dt.date.today(),
            display_format="DD/MM/YYYY"
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

# --- helpers: mappings---
DAY_MAP = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
MONTH_MAP = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}

def featurize(df):
    df = df.copy()
    df['day_of_week'] = df['fecha'].dt.day_name().map(DAY_MAP)
    df['month'] = df['fecha'].dt.month_name().map(MONTH_MAP)
    df['year'] = df['fecha'].dt.year
    df['is_fornight'] = df['fecha'].dt.day.isin([14,15]).astype(int)

    df = df.sort_values(['producto','fecha']).reset_index(drop=True)

    # Lags
    df['lag_1']  = df.groupby('producto')['venta'].shift(1)
    df['lag_7']  = df.groupby('producto')['venta'].shift(7)
    df['lag_30'] = df.groupby('producto')['venta'].shift(30)
    df['lag_60'] = df.groupby('producto')['venta'].shift(60)

    # Diferencias
    df['diff_1'] = df.groupby('producto')['venta'].diff(1)
    df['diff_7'] = df.groupby('producto')['venta'].diff(7)

    # Rolling features

    df['rolling_mean_15'] = (df.groupby('producto')['venta']
                               .transform(lambda x: x.rolling(15, min_periods=1).mean().shift(1)))
    df['last_30_day_avg'] = (df.groupby('producto')['venta']
                               .transform(lambda x: x.rolling(30, min_periods=1).mean().shift(1)))
    df['rolling_std_15'] = (df.groupby('producto')['venta']
                               .transform(lambda x: x.rolling(15, min_periods=1).std().shift(1)))
    df['rolling_std_7'] = (df.groupby('producto')['venta']
                         .transform(lambda x: x.rolling(7, min_periods=1).std().shift(1)))
    df['rolling_max_15'] = (df.groupby('producto')['venta']
                               .transform(lambda x: x.rolling(15, min_periods=1).max().shift(1)))
    df['rolling_min_15'] = (df.groupby('producto')['venta']
                               .transform(lambda x: x.rolling(15, min_periods=1).min().shift(1)))
    
    # Promedio dias iguales
    def avg_last_4_same_day(series, dates):
        out = []
        for i in range(len(series)):
            mask = dates.dt.day_name() == dates.iloc[i].day_name()
            prev = series[mask].iloc[:i]
            out.append(prev.iloc[-4:].mean() if len(prev)>=4 else (prev.mean() if len(prev)>0 else np.nan))
        return out
    
    df['avg_last_4_same_day'] = avg_last_4_same_day(df['venta'], df['fecha'])

    return df

PRODUCTS = ["Producto 1","Producto 2","Producto 3","Producto 4"]

# Traer los datos reales
# -----------------------------
def get_real_data(selected_date: dt.date) -> pd.DataFrame:

    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    models_path = os.path.join(base_path, "models")
    raw_path = os.path.join(base_path, "raw", "data_proyecto.csv")

    productos = ["Producto 1", "Producto 2", "Producto 3", "Producto 4"]
    resultados = []

    for producto in productos:
        suf = producto.lower().replace(" ", "")  # "Producto 1" -> "producto1"
        model_path = os.path.join(models_path, f"rf_{suf}.pkl")
        features_path = os.path.join(models_path, f"features_{suf}.json")

        print(f"Cargando modelo: {model_path}")  # 👈 agrega esta línea

        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"⚠️ Error cargando {model_path}: {e}")
            raise

        # carga modelo y features
        model = joblib.load(model_path)
        with open(features_path) as f:
            feats = json.load(f)

        # carga historico
        hist = pd.read_csv(raw_path, parse_dates=["fecha"])
        hist = hist[hist["producto"] == producto].sort_values("fecha").reset_index(drop=True)

        # genera predicciones hasta la fecha seleccionada 
        data = hist[hist["fecha"].dt.date <= selected_date].copy()
        data = featurize(data)
        X = data[feats].copy()
        mask = X.notna().all(axis=1)
        preds = pd.Series(np.nan, index=data.index)
        preds[mask] = model.predict(X[mask])

        resid = data.loc[mask, "venta"] - preds[mask]
        sigma = resid.std() if resid.notna().any() else 0.0
        ci = 1.96 * (sigma if sigma > 0 else 1.0)

        out = pd.DataFrame({
            "fecha": data["fecha"],
            "producto": data["producto"],
            "venta_real": data["venta"].values,
            "venta_predicha": preds.values,
        })
        out["fondo_recomendado"] = np.where(out["venta_predicha"].notna(),
                                            np.maximum(out["venta_predicha"]*0.95, 3e5),
                                            np.nan)
        out["ci_inferior"] = out["venta_predicha"] - ci
        out["ci_superior"] = out["venta_predicha"] + ci
        out["recurso_sobrante"] = np.where(out["venta_predicha"].notna(),
                                           np.maximum(out["venta_predicha"] - out["venta_real"], 0),
                                           np.nan)
        resultados.append(out)

    # --- une todo ---
    df_final = pd.concat(resultados, ignore_index=True)
    return df_final

def format_currency(x):
    return "" if pd.isna(x) else f"${x:,.0f}".replace(",", ".")

def compute_kpis(df: pd.DataFrame, date: dt.date):
    dfd = df[df["fecha"].dt.date == date]
    fondo_total = dfd["fondo_recomendado"].sum()
    rmse = np.sqrt(np.mean((dfd["venta_real"] - dfd["venta_predicha"])**2))
    var_pct = np.random.uniform(-0.15, 0.15)  # solo decorativo
    return fondo_total, rmse, var_pct

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
    df = get_real_data(date)

    # update de los kpis
    fondo_total, rmse, var_pct = compute_kpis(df, date)
    k1 = format_currency(fondo_total)
    k2 = format_currency(rmse)
    k3 = f"{var_pct*100:.1f}%" if var_pct is not None else "—"

    # update de la grafica de linea
    series = df.groupby("fecha")[["venta_real","venta_predicha"]].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series["fecha"], 
        y=series["venta_real"], 
        name="Venta real", 
        mode="lines+markers",
        line=dict(color="#9287f9")
        ))
    fig.add_trace(go.Scatter(
        x=series["fecha"], 
        y=series["venta_predicha"], 
        name="Venta predicha", 
        mode="lines+markers",
        line=dict(color="#a6c0ed")
        ))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                      hovermode="x unified",
                      legend_title_text="",
                      height=300,
                      font=dict(family="Segoe UI, Segoe UI Variable, -apple-system, system-ui, Arial, sans-serif"),
                      plot_bgcolor="white",   # fondo del área de la gráfica
                      paper_bgcolor="white",  # fondo general
                      xaxis=dict(
                          showgrid=False,      # sin líneas verticales
                          zeroline=False,
                          linecolor="#dee2e6",
                          tickfont=dict(color="#495057")
    ),
    yaxis=dict(
        showgrid=True,       # activa líneas horizontales
        gridcolor="#e9ecef", # gris claro (puedes usar #dee2e6 o #f1f3f5 si quieres más suave)
        zeroline=False,
        linecolor="#dee2e6",
        tickfont=dict(color="#495057")
        )
    )

    # update de la tabla
    sumd = (df[df["fecha"].dt.date == date]
            .groupby("producto", as_index=False)
            .agg(fondo_recomendado=("fondo_recomendado","sum"),
                 recurso_sobrante=("recurso_sobrante","sum"),
                 prediccion=("venta_predicha","sum"),
                 venta_real=("venta_real","sum")))

    for c in ["fondo_recomendado","recurso_sobrante","prediccion","venta_real"]:
        sumd[c] = sumd[c].map(format_currency)

    columns = [{"name": col.replace("_", " ").title(), "id": col} for col in sumd.columns]
    data = sumd.to_dict("records")

    return k1, k2, k3, fig, columns, data


# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)