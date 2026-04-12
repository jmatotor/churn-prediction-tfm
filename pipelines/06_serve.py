"""
pipelines/06_serve.py
======================
Arranca la API REST de predicción de churn.
Uso: python pipelines/06_serve.py

Una vez arrancada:
  - Docs interactivas : http://localhost:8000/docs
  - Health check      : http://localhost:8000/health
  - Predicción        : POST http://localhost:8000/predict
  - Batch             : POST http://localhost:8000/predict/batch

Ejemplo de petición con curl:
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "gender": 0,
      "senior_citizen": 0,
      "partner": 0,
      "dependents": 0,
      "tenure_months": 2,
      "phone_service": 1,
      "multiple_lines": 0,
      "online_security": 0,
      "online_backup": 0,
      "device_protection": 0,
      "tech_support": 0,
      "streaming_tv": 0,
      "streaming_movies": 0,
      "contract": 0,
      "paperless_billing": 1,
      "monthly_charges": 70.0,
      "total_charges": 140.0,
      "internet_service_Fiber optic": 1,
      "internet_service_No": 0,
      "payment_method_Credit card (automatic)": 0,
      "payment_method_Electronic check": 1,
      "payment_method_Mailed check": 0,
      "n_services": 0,
      "is_month_to_month": 1,
      "charges_per_tenure": 23.3,
      "has_fiber": 1,
      "is_new_customer": 1,
      "is_loyal_customer": 0
    }'
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from serve import run_server

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

HOST   = "0.0.0.0"
PORT   = 8000
RELOAD = False   # True para desarrollo (recarga automática al editar serve.py)

# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    model_path = ROOT / "models" / "best_model.pkl"

    if not model_path.exists():
        print("ERROR: No se encuentra best_model.pkl")
        print("Ejecuta primero: python pipelines/04_train.py")
        sys.exit(1)

    print(f"Arrancando API en http://{HOST}:{PORT}")
    print(f"Documentación:   http://localhost:{PORT}/docs")
    print(f"Health check:    http://localhost:{PORT}/health")
    print("Ctrl+C para parar.\n")

    run_server(host=HOST, port=PORT, reload=RELOAD)