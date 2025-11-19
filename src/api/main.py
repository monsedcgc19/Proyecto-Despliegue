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
    grupo = df.groupby('producto')['venta']
    df['lag_1'] = grupo.shift(1)
    df['lag_2'] = grupo.shift(2)          # ← nuevo
    df['lag_7'] = grupo.shift(7)
    df['lag_8'] = grupo.shift(8)          # ← nuevo
    df['lag_30'] = grupo.shift(30)
    df['lag_60'] = grupo.shift(60)
    # -----------------------------------------
    # Diferencias (diff) sin usar la venta actual
    # -----------------------------------------
    # diff_1: diferencia entre el penúltimo y el antepenúltimo valor
    df['diff_1'] = df['lag_1'] - df['lag_2']
    # diff_7: cambio en 7 días usando solo lags
    # (venta t-1 vs venta t-8)
    df['diff_7'] = df['lag_1'] - df['lag_8']
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
    """
    Devuelve ventas reales + predicciones para los 4 productos,
    incluyendo predicción hacia adelante hasta fecha_fin (máx. +3 semanas).
    """
    hist = load_hist_data()  # columnas esperadas: fecha, producto, venta
    hist = hist.sort_values(["producto", "fecha"]).reset_index(drop=True)
    # Fecha máxima real en el histórico (global)
    last_real_global = hist["fecha"].max().date()
    max_allowed_date = last_real_global + dt.timedelta(weeks=3)
    # Limitar la fecha de fin a máximo +3 semanas
    if fecha_fin > max_allowed_date:
        fecha_fin = max_allowed_date
    resultados = []
    for producto in PRODUCTOS:
        # Histórico solo de ese producto
        data_p = hist[hist["producto"] == producto].copy()
        if data_p.empty:
            continue
        # Última fecha real de ese producto
        last_real_prod = data_p["fecha"].max().date()
        # Fecha objetivo (ya limitada globalmente)
        fecha_objetivo = fecha_fin
        # ---------------------------
        # 1) Parte histórica (<= min(fecha_objetivo, last_real_prod))
        # ---------------------------
        fecha_corte_hist = min(fecha_objetivo, last_real_prod)
        data_hist = data_p[data_p["fecha"].dt.date <= fecha_corte_hist].copy()
        if data_hist.empty:
            continue
        # Features históricas
        data_hist_feat = featurize(data_hist)
        # Modelo y lista de features
        model, feats = load_model_and_features(producto)
        X_hist = data_hist_feat[feats].copy()
        mask_hist = X_hist.notna().all(axis=1)
        preds_hist = pd.Series(np.nan, index=data_hist_feat.index)
        preds_hist[mask_hist] = model.predict(X_hist[mask_hist])
        # Desviación estándar de residuos para construir intervalo de confianza
        resid = data_hist_feat.loc[mask_hist, "venta"] - preds_hist[mask_hist]
        sigma = resid.std() if resid.notna().any() else 0.0
        ci = 1.96 * (sigma if sigma > 0 else 1.0)
        out_hist = pd.DataFrame({
            "fecha": data_hist_feat["fecha"].values,
            "producto": data_hist_feat["producto"].values,
            "venta_real": data_hist_feat["venta"].values,
            "venta_predicha": preds_hist.values,
        })
        # ---------------------------
        # 2) Parte futura (> last_real_prod hasta fecha_objetivo)
        #    Predicción recursiva día a día
        # ---------------------------
        future_rows = []
        if fecha_objetivo > last_real_prod:
            start_future = pd.to_datetime(last_real_prod) + dt.timedelta(days=1)
            end_future = pd.to_datetime(fecha_objetivo)
            future_dates = pd.date_range(start_future, end_future, freq="D")
            # Base para features futuras: histórico con columna 'venta'
            base_for_feat = data_hist[["fecha", "producto", "venta"]].copy()
            for f_date in future_dates:
                # Agregar fila vacía (venta desconocida)
                new_row = pd.DataFrame({
                    "fecha": [f_date],
                    "producto": [producto],
                    "venta": [np.nan],
                })
                base_for_feat = pd.concat([base_for_feat, new_row], ignore_index=True)
                # Recalcular features con histórico + fila nueva
                feat_all = featurize(base_for_feat.copy())
                row_feat = feat_all[
                    (feat_all["producto"] == producto) &
                    (feat_all["fecha"] == f_date)
                ]
                if row_feat.empty:
                    pred_val = np.nan
                else:
                    X_f = row_feat[feats]
                    mask_f = X_f.notna().all(axis=1)
                    if mask_f.iloc[0]:
                        pred_val = float(model.predict(X_f)[0])
                    else:
                        pred_val = np.nan
                # Actualizar la serie de 'venta' con la predicción
                base_for_feat.loc[
                    (base_for_feat["producto"] == producto) &
                    (base_for_feat["fecha"] == f_date),
                    "venta"
                ] = pred_val
                # Guardar fila futura para salida de la API
                future_rows.append({
                    "fecha": f_date,
                    "producto": producto,
                    "venta_real": np.nan,
                    "venta_predicha": pred_val,
                })
        if future_rows:
            out_future = pd.DataFrame(future_rows)
            out_prod = pd.concat([out_hist, out_future], ignore_index=True)
        else:
            out_prod = out_hist
        # ---------------------------
        # 3) Columnas derivadas (igual que antes)
        # ---------------------------
        out_prod["fondo_recomendado"] = np.where(
            out_prod["venta_predicha"].notna(),
            np.maximum(out_prod["venta_predicha"] * 0.95, 3e5),
            np.nan,
        )
        out_prod["ci_inferior"] = out_prod["venta_predicha"] - ci
        out_prod["ci_superior"] = out_prod["venta_predicha"] + ci
        out_prod["recurso_sobrante"] = np.where(
            out_prod["venta_predicha"].notna() & out_prod["venta_real"].notna(),
            np.maximum(out_prod["venta_predicha"] - out_prod["venta_real"], 0),
            np.nan,
        )
        resultados.append(out_prod)
    # Si no hay datos, devolver df vacío con las columnas esperadas
    if not resultados:
        return pd.DataFrame(columns=[
            "fecha", "producto", "venta_real", "venta_predicha",
            "fondo_recomendado", "ci_inferior", "ci_superior", "recurso_sobrante",
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