# src/api/main.py
# pip install fastapi uvicorn joblib pandas numpy
import json
import joblib
import datetime as dt
from functools import lru_cache
from typing import List
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
# -----------------------------
# Config
# -----------------------------
MODELS_DIR = "models"
DATA_PATH = "data/raw/data_proyecto.csv"   # ajusta si tu csv está en otro lado
PRODUCTOS = ["Producto 1", "Producto 2", "Producto 3", "Producto 4"]
DAY_MAP = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
MONTH_MAP = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
             'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
# -----------------------------
# Esquemas de entrada / salida
# -----------------------------
class PredictRequest(BaseModel):
    fecha_fin: dt.date   # fecha seleccionada en el dashboard (YYYY-MM-DD)
class PredRow(BaseModel):
    fecha: dt.datetime
    producto: str
    venta_real: float | None
    venta_predicha: float | None
    fondo_recomendado: float | None
    ci_inferior: float | None
    ci_superior: float | None
    recurso_sobrante: float | None
class PredictResponse(BaseModel):
    predicciones: List[PredRow]
# -----------------------------
# Helpers
# -----------------------------
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # -----------------------------------------
    # Variables de calendario
    # -----------------------------------------
    df['day_of_week'] = df['fecha'].dt.day_name().map(DAY_MAP)
    df['month'] = df['fecha'].dt.month_name().map(MONTH_MAP)
    df['year'] = df['fecha'].dt.year   # ← agregado
    df['is_fornight'] = df['fecha'].dt.day.isin([14, 15]).astype(int)
    # -----------------------------------------
    # Orden por producto y fecha
    # -----------------------------------------
    df = df.sort_values(['producto','fecha']).reset_index(drop=True)
    # -----------------------------------------
    # Lags
    # -----------------------------------------
    df['lag_1'] = df.groupby('producto')['venta'].shift(1)      # ← agregado
    df['lag_7'] = df.groupby('producto')['venta'].shift(7)
    df['lag_30'] = df.groupby('producto')['venta'].shift(30)
    df['lag_60'] = df.groupby('producto')['venta'].shift(60)
    # -----------------------------------------
    # Diferencias (diff)
    # -----------------------------------------
    df['diff_1'] = df.groupby('producto')['venta'].diff(1)      # ← agregado
    df['diff_7'] = df.groupby('producto')['venta'].diff(7)      # ← agregado
    # -----------------------------------------
    # Promedios móviles y rolling stats
    # -----------------------------------------
    df['rolling_mean_15'] = (
        df.groupby('producto')['venta']
        .transform(lambda x: x.rolling(15, min_periods=1).mean().shift(1))
    )
    df['rolling_std_7'] = (                                       # ← agregado
        df.groupby('producto')['venta']
        .transform(lambda x: x.rolling(7, min_periods=1).std().shift(1))
    )
    df['rolling_std_15'] = (
        df.groupby('producto')['venta']
        .transform(lambda x: x.rolling(15, min_periods=1).std().shift(1))
    )
    df['rolling_max_15'] = (
        df.groupby('producto')['venta']
        .transform(lambda x: x.rolling(15, min_periods=1).max().shift(1))
    )
    df['rolling_min_15'] = (
        df.groupby('producto')['venta']
        .transform(lambda x: x.rolling(15, min_periods=1).min().shift(1))
    )
    # -----------------------------------------
    # Promedio de los últimos 4 días iguales
    # -----------------------------------------
    def avg_last_4_same_day(series, dates):
        out = []
        for i in range(len(series)):
            mask = dates.dt.day_name() == dates.iloc[i].day_name()
            prev = series[mask].iloc[:i]
            if len(prev) >= 4:
                out.append(prev.iloc[-4:].mean())
            elif len(prev) > 0:
                out.append(prev.mean())
            else:
                out.append(np.nan)
        return out
    df['avg_last_4_same_day'] = avg_last_4_same_day(df['venta'], df['fecha'])
    # -----------------------------------------
    # Últimos 30 días promedio
    # -----------------------------------------
    df['last_30_day_avg'] = (
        df.groupby('producto')['venta']
        .transform(lambda x: x.rolling(30, min_periods=1).mean().shift(1))
    )
    return df

@lru_cache(maxsize=16)
def load_hist_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=['fecha'])
    return df
@lru_cache(maxsize=16)
def load_model_and_features(producto: str):
    suf = producto.lower().replace(" ", "")   # "Producto 1" -> "producto1"
    model_path = f"{MODELS_DIR}/rf_{suf}.pkl"
    feats_path = f"{MODELS_DIR}/features_{suf}.json"
    model = joblib.load(model_path)
    with open(feats_path) as f:
        feats = json.load(f)
    return model, feats
def get_real_data(fecha_fin: dt.date) -> pd.DataFrame:
    """Equivalente a tu get_real_data, pero para los 4 productos."""
    hist = load_hist_data()
    resultados = []
    for producto in PRODUCTOS:
        model, feats = load_model_and_features(producto)
        data_p = hist[hist['producto'] == producto].copy()
        data_p = data_p[data_p['fecha'].dt.date <= fecha_fin]
        if data_p.empty:
            continue
        data_p = featurize(data_p)
        X = data_p[feats].copy()
        mask = X.notna().all(axis=1)
        preds = pd.Series(np.nan, index=data_p.index)
        preds[mask] = model.predict(X[mask])
        resid = data_p.loc[mask, 'venta'] - preds[mask]
        sigma = resid.std() if resid.notna().any() else 0.0
        ci = 1.96 * (sigma if sigma > 0 else 1.0)
        out = pd.DataFrame({
            "fecha": data_p["fecha"],
            "producto": data_p["producto"],
            "venta_real": data_p["venta"].values,
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
    if not resultados:
        return pd.DataFrame(columns=[
            "fecha","producto","venta_real","venta_predicha",
            "fondo_recomendado","ci_inferior","ci_superior","recurso_sobrante"
        ])
    df_final = pd.concat(resultados, ignore_index=True).sort_values("fecha")
    return df_final
# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="API Predicción de Ventas")

@app.get("/health")
def health():
    return {"status": "ok"}
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = get_real_data(req.fecha_fin)
    # :small_blue_diamond: Convertir NaN / inf a None para que JSON los acepte
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    records = df.to_dict(orient="records")
    return {"predicciones": records}