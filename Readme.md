# Churn Prediction TFM

Sistema de predicción de abandono de clientes (churn) end-to-end, diseñado e implementado como Trabajo Fin de Máster con un enfoque de ingeniería de datos real.

## Descripción

El proyecto implementa un pipeline completo de Machine Learning siguiendo una arquitectura medallion (Bronze → Silver → Gold), desde la ingesta del dato raw hasta el serving del modelo mediante una API REST. El objetivo es predecir qué clientes de una compañía de telecomunicaciones tienen mayor probabilidad de abandonar el servicio.

## Arquitectura

```
Bronze (raw)  →  Silver (limpio)  →  Gold (features)  →  Modelo  →  API REST
     ↓                ↓                    ↓               ↓
  CSV raw          Parquet             Parquet x4        .pkl       FastAPI
                  tipado            train/test split    MLflow      /predict
```

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Procesamiento | Python, Pandas, PySpark |
| Almacenamiento | Delta Lake / Parquet |
| Modelado | XGBoost, LightGBM, Scikit-learn |
| Hiperparámetros | Optuna |
| Experiment tracking | MLflow |
| Explicabilidad | SHAP |
| Serving | FastAPI + Uvicorn |
| Cloud | Azure Data Factory, ADLS Gen2 |

## Estructura del repositorio

```
churn-prediction-tfm/
├── data/
│   ├── bronze/                  ← raw_data.csv (Telco Customer Churn)
│   ├── silver/                  ← customers_clean.parquet
│   └── gold/                    ← X_train, X_test, y_train, y_test
├── src/
│   ├── bronze_to_silver.py      ← limpieza y validación
│   ├── eda_silver.py            ← análisis exploratorio
│   ├── silver_to_gold.py        ← feature engineering
│   ├── train.py                 ← entrenamiento y selección de modelo
│   ├── evaluate.py              ← métricas y SHAP
│   ├── feature_experiments.py   ← experimentos de features
│   └── serve.py                 ← API REST
├── pipelines/
│   ├── 01_bronze_to_silver.py
│   ├── 02_eda_silver.py
│   ├── 03_silver_to_gold.py
│   ├── 04_train.py
│   ├── 05_evaluate.py
│   ├── 05b_feature_experiments.py
│   └── 06_serve.py
├── models/                      ← best_model.pkl
├── reports/
│   ├── eda/                     ← gráficos del EDA
│   └── evaluation/              ← curvas ROC, SHAP, matriz de confusión
├── mlruns/                      ← experimentos MLflow
└── requirements.txt
```

## Dataset

**Telco Customer Churn** (IBM / Kaggle)
- 7.043 clientes, 21 variables originales
- Churn rate: 26.5% (dataset desbalanceado)
- Fuente: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Instalación

```bash
git clone https://github.com/tu-usuario/churn-prediction-tfm
cd churn-prediction-tfm
pip install -r requirements.txt
```

## Ejecución del pipeline completo

```bash
# 1. Bronze → Silver (limpieza)
python pipelines/01_bronze_to_silver.py

# 2. EDA sobre Silver
python pipelines/02_eda_silver.py

# 3. Silver → Gold (feature engineering)
python pipelines/03_silver_to_gold.py

# 4. Entrenamiento (4 modelos + Optuna + MLflow)
python pipelines/04_train.py

# 5. Evaluación + SHAP
python pipelines/05_evaluate.py

# 5b. Experimentos de selección de features
python pipelines/05b_feature_experiments.py

# 6. Arrancar la API
python pipelines/06_serve.py
```

## API REST

Una vez arrancada la API:

| Endpoint | Método | Descripción |
|---|---|---|
| `/health` | GET | Estado de la API y modelo cargado |
| `/model/info` | GET | Features y configuración del modelo |
| `/predict` | POST | Predicción individual con razones SHAP |
| `/predict/batch` | POST | Predicción para múltiples clientes |
| `/docs` | GET | Documentación interactiva (Swagger) |

**Ejemplo de predicción:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 2,
    "contract": 0,
    "monthly_charges": 70.0,
    "total_charges": 140.0,
    "internet_service_Fiber optic": 1,
    "payment_method_Electronic check": 1,
    "is_month_to_month": 1,
    "charges_per_tenure": 23.3,
    "has_fiber": 1,
    "is_new_customer": 1,
    "gender": 0, "senior_citizen": 0, "partner": 0, "dependents": 0,
    "phone_service": 1, "multiple_lines": 0, "online_security": 0,
    "online_backup": 0, "device_protection": 0, "tech_support": 0,
    "streaming_tv": 0, "streaming_movies": 0, "paperless_billing": 1,
    "internet_service_No": 0,
    "payment_method_Credit card (automatic)": 0,
    "payment_method_Mailed check": 0,
    "n_services": 0, "is_loyal_customer": 0
  }'
```

**Respuesta:**

```json
{
  "churn_probability": 0.847,
  "churn_prediction": true,
  "risk_level": "muy_alto",
  "top_reasons": [
    {"feature": "contract", "value": 0, "shap_value": 0.42, "direction": "aumenta_churn"},
    {"feature": "charges_per_tenure", "value": 23.3, "shap_value": 0.31, "direction": "aumenta_churn"},
    {"feature": "tenure_months", "value": 2, "shap_value": 0.18, "direction": "aumenta_churn"}
  ],
  "model_name": "xgboost"
}
```

## Experimentos registrados en MLflow

```bash
mlflow ui --port 5000
# Abrir http://localhost:5000
```

Experimentos disponibles:
- `churn-prediction-tfm`: comparativa de 4 modelos (LR, RF, XGBoost, LightGBM)
- `churn-feature-experiments`: 5 configuraciones de features comparadas

## Resultados

| Modelo | ROC-AUC | F1 | Recall |
|---|---|---|---|
| XGBoost (ganador) | 0.847 | 0.639 | 0.800 |
| LightGBM | — | — | — |
| Random Forest | — | — | — |
| Logistic Regression | — | — | — |

*Completar con los resultados reales del pipeline 04.*

**Top features por importancia SHAP:**
1. `contract` (tipo de contrato)
2. `charges_per_tenure` (coste relativo a la antigüedad — feature derivada)
3. `is_month_to_month` (flag contrato mensual)
4. `internet_service_Fiber optic`
5. `monthly_charges`

## Decisiones de diseño destacadas

**ELT sobre ETL:** los datos se cargan en Bronze sin transformar (fuente de verdad) y se transforman posteriormente en Silver. Esto permite regenerar cualquier capa sin perder el dato original.

**Selección de modelo automatizada:** el pipeline entrena 4 modelos distintos con búsqueda de hiperparámetros via Optuna y selecciona automáticamente el ganador por ROC-AUC. El modelo ganador se registra en MLflow con tag `best_model=true`.

**Explicabilidad como requisito:** cada predicción de la API incluye las top 3 razones calculadas con SHAP, no solo el score. Esto hace el modelo accionable para negocio.

**Experimentos de features trazables:** el paso 05b demuestra el impacto de cada decisión de feature engineering de forma cuantitativa y registrada en MLflow.

## Autor

Jesús María Matos Torres — Trabajo Fin de Máster