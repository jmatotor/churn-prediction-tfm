"""
src/serve.py
============
API REST para predicción de churn en tiempo real.

Endpoints:
  GET  /health          - Estado de la API y modelo cargado
  GET  /model/info      - Información del modelo en producción
  POST /predict         - Predicción para un cliente individual
  POST /predict/batch   - Predicción para múltiples clientes

El modelo se carga una sola vez al arrancar la API (patrón lifespan).
Las features de entrada son las mismas que X_train del Gold.
La respuesta incluye el score de churn y el top 3 de razones (SHAP).
"""

import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PATHS (relativos a la raíz del proyecto)
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
GOLD_DIR   = ROOT / "data" / "gold"

# ---------------------------------------------------------------------------
# ESTADO GLOBAL DE LA API
# ---------------------------------------------------------------------------

# Se inicializa en el lifespan y se usa en los endpoints
app_state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# LIFESPAN — carga del modelo al arrancar
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga el modelo y el explainer SHAP una sola vez al arrancar.
    Se libera al apagar la API.
    """
    logger.info("Cargando modelo...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encuentra el modelo en {MODEL_PATH}. "
            "Ejecuta primero: python pipelines/04_train.py"
        )

    artifact   = joblib.load(MODEL_PATH)
    model      = artifact["model"]
    model_name = artifact["model_name"]

    # Cargar nombres de features desde Gold
    X_sample = pd.read_parquet(GOLD_DIR / "X_test.parquet").head(100)
    feature_names = X_sample.columns.tolist()

    # Inicializar explainer SHAP
    logger.info("Inicializando SHAP explainer...")
    if model_name in ("xgboost", "lightgbm", "random_forest"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_sample)

    app_state["model"]         = model
    app_state["model_name"]    = model_name
    app_state["feature_names"] = feature_names
    app_state["explainer"]     = explainer

    logger.info(f"Modelo '{model_name}' cargado con {len(feature_names)} features.")
    logger.info("API lista.")

    yield

    # Cleanup al apagar
    app_state.clear()
    logger.info("API apagada.")


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Churn Prediction API",
    description = "Predicción de churn de clientes con explicabilidad SHAP.",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ---------------------------------------------------------------------------
# SCHEMAS DE ENTRADA
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """
    Features de un cliente para predecir su probabilidad de churn.
    Corresponden exactamente a las columnas del dataset Gold.
    Los valores por defecto son las medianas/modas del dataset Telco.
    """
    gender:                              int   = Field(0,    ge=0, le=1,   description="0=Female, 1=Male")
    senior_citizen:                      int   = Field(0,    ge=0, le=1,   description="0=No, 1=Sí")
    partner:                             int   = Field(0,    ge=0, le=1,   description="0=No, 1=Sí")
    dependents:                          int   = Field(0,    ge=0, le=1,   description="0=No, 1=Sí")
    tenure_months:                       int   = Field(32,   ge=0, le=120, description="Meses como cliente")
    phone_service:                       int   = Field(1,    ge=0, le=1)
    multiple_lines:                      int   = Field(0,    ge=0, le=1)
    online_security:                     int   = Field(0,    ge=0, le=1)
    online_backup:                       int   = Field(0,    ge=0, le=1)
    device_protection:                   int   = Field(0,    ge=0, le=1)
    tech_support:                        int   = Field(0,    ge=0, le=1)
    streaming_tv:                        int   = Field(0,    ge=0, le=1)
    streaming_movies:                    int   = Field(0,    ge=0, le=1)
    contract:                            int   = Field(0,    ge=0, le=2,   description="0=Month-to-month, 1=One year, 2=Two year")
    paperless_billing:                   int   = Field(1,    ge=0, le=1)
    monthly_charges:                     float = Field(64.0, ge=0.0,       description="Cargo mensual en $")
    total_charges:                       float = Field(1397.0, ge=0.0,     description="Cargo total acumulado en $")
    internet_service_Fiber_optic:        int   = Field(0,    ge=0, le=1,   alias="internet_service_Fiber optic")
    internet_service_No:                 int   = Field(0,    ge=0, le=1)
    payment_method_Credit_card:          int   = Field(0,    ge=0, le=1,   alias="payment_method_Credit card (automatic)")
    payment_method_Electronic_check:     int   = Field(0,    ge=0, le=1,   alias="payment_method_Electronic check")
    payment_method_Mailed_check:         int   = Field(0,    ge=0, le=1,   alias="payment_method_Mailed check")
    n_services:                          int   = Field(0,    ge=0, le=6,   description="Número de servicios contratados")
    is_month_to_month:                   int   = Field(1,    ge=0, le=1)
    charges_per_tenure:                  float = Field(2.0,  ge=0.0,       description="monthly_charges / (tenure_months + 1)")
    has_fiber:                           int   = Field(0,    ge=0, le=1)
    is_new_customer:                     int   = Field(0,    ge=0, le=1,   description="tenure_months < 12")
    is_loyal_customer:                   int   = Field(0,    ge=0, le=1,   description="tenure_months > 48")

    model_config = {"populate_by_name": True}


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


# ---------------------------------------------------------------------------
# SCHEMAS DE SALIDA
# ---------------------------------------------------------------------------

class ShapReason(BaseModel):
    feature:    str
    value:      float
    shap_value: float
    direction:  str   # "aumenta_churn" | "reduce_churn"


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction:  bool
    risk_level:        str          # "bajo" | "medio" | "alto" | "muy_alto"
    top_reasons:       list[ShapReason]
    model_name:        str


class BatchResponse(BaseModel):
    predictions:    list[PredictionResponse]
    total_customers: int
    churn_predicted: int
    avg_probability: float


class HealthResponse(BaseModel):
    status:     str
    model_name: str
    n_features: int


class ModelInfoResponse(BaseModel):
    model_name:   str
    feature_names: list[str]
    n_features:   int
    threshold:    float
    description:  str


# ---------------------------------------------------------------------------
# LÓGICA DE PREDICCIÓN
# ---------------------------------------------------------------------------

RISK_THRESHOLDS = {
    "bajo":     (0.0,  0.3),
    "medio":    (0.3,  0.5),
    "alto":     (0.5,  0.7),
    "muy_alto": (0.7,  1.01),
}

DEFAULT_THRESHOLD = 0.5


def _risk_level(prob: float) -> str:
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= prob < high:
            return level
    return "muy_alto"


def _customer_to_df(customer: CustomerFeatures, feature_names: list[str]) -> pd.DataFrame:
    """
    Convierte un CustomerFeatures en un DataFrame con las columnas en el
    orden correcto que espera el modelo.
    """
    # Mapeo alias → nombre real de columna
    alias_map = {
        "internet_service_Fiber_optic":    "internet_service_Fiber optic",
        "payment_method_Credit_card":      "payment_method_Credit card (automatic)",
        "payment_method_Electronic_check": "payment_method_Electronic check",
        "payment_method_Mailed_check":     "payment_method_Mailed check",
    }

    raw = customer.model_dump(by_alias=True)

    # Renombrar campos con alias
    row = {}
    for k, v in raw.items():
        row[alias_map.get(k, k)] = v

    df = pd.DataFrame([row])

    # Reordenar para que coincida con el orden del modelo
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Features faltantes en la petición: {missing}",
        )

    return df[feature_names]


def _get_shap_reasons(
    explainer,
    df_input:      pd.DataFrame,
    feature_names: list[str],
    model_name:    str,
    top_n:         int = 3,
) -> list[ShapReason]:
    """
    Calcula SHAP values y devuelve las top_n razones más influyentes.
    """
    shap_vals = explainer.shap_values(df_input)

    # Random Forest devuelve lista [clase0, clase1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_row = shap_vals[0]

    # Ordenar por valor absoluto descendente
    sorted_idx = np.argsort(np.abs(shap_row))[::-1][:top_n]

    reasons = []
    for idx in sorted_idx:
        reasons.append(ShapReason(
            feature    = feature_names[idx],
            value      = float(df_input.iloc[0, idx]),
            shap_value = float(shap_row[idx]),
            direction  = "aumenta_churn" if shap_row[idx] > 0 else "reduce_churn",
        ))

    return reasons


def _predict_one(
    customer:      CustomerFeatures,
    model,
    explainer,
    feature_names: list[str],
    model_name:    str,
) -> PredictionResponse:
    df_input = _customer_to_df(customer, feature_names)

    prob       = float(model.predict_proba(df_input)[0, 1])
    prediction = prob >= DEFAULT_THRESHOLD
    risk       = _risk_level(prob)
    reasons    = _get_shap_reasons(explainer, df_input, feature_names, model_name)

    return PredictionResponse(
        churn_probability = round(prob, 4),
        churn_prediction  = prediction,
        risk_level        = risk,
        top_reasons       = reasons,
        model_name        = model_name,
    )


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    """Comprueba que la API está activa y el modelo cargado."""
    if not app_state:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    return HealthResponse(
        status     = "ok",
        model_name = app_state["model_name"],
        n_features = len(app_state["feature_names"]),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Sistema"])
def model_info():
    """Devuelve información sobre el modelo activo."""
    if not app_state:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    return ModelInfoResponse(
        model_name    = app_state["model_name"],
        feature_names = app_state["feature_names"],
        n_features    = len(app_state["feature_names"]),
        threshold     = DEFAULT_THRESHOLD,
        description   = "Modelo de predicción de churn entrenado con Optuna y registrado en MLflow.",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(customer: CustomerFeatures):
    """
    Predice la probabilidad de churn para un cliente individual.
    Devuelve el score, nivel de riesgo y top 3 razones con SHAP.
    """
    if not app_state:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    try:
        return _predict_one(
            customer      = customer,
            model         = app_state["model"],
            explainer     = app_state["explainer"],
            feature_names = app_state["feature_names"],
            model_name    = app_state["model_name"],
        )
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predicción"])
def predict_batch(request: BatchRequest):
    """
    Predicción para múltiples clientes (máximo 1000 por petición).
    """
    if not app_state:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    predictions = []
    for customer in request.customers:
        try:
            pred = _predict_one(
                customer      = customer,
                model         = app_state["model"],
                explainer     = app_state["explainer"],
                feature_names = app_state["feature_names"],
                model_name    = app_state["model_name"],
            )
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Error en cliente del batch: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    probs = [p.churn_probability for p in predictions]

    return BatchResponse(
        predictions     = predictions,
        total_customers = len(predictions),
        churn_predicted = sum(p.churn_prediction for p in predictions),
        avg_probability = round(float(np.mean(probs)), 4),
    )


# ---------------------------------------------------------------------------
# ARRANQUE DIRECTO
# ---------------------------------------------------------------------------

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    uvicorn.run(
        "serve:app",
        host    = host,
        port    = port,
        reload  = reload,
        app_dir = str(Path(__file__).parent),
    )


if __name__ == "__main__":
    run_server()