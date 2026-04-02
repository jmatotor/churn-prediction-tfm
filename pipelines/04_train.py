"""
pipelines/04_train.py
======================
Ejecuta el entrenamiento del modelo de churn.
Uso: python pipelines/04_train.py

Para ver los experimentos después:
  mlflow ui --port 5000
  Abrir http://localhost:5000
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from train import run_training

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

GOLD_DIR    = ROOT / "data" / "gold"
MODELS_DIR  = ROOT / "models"
MLFLOW_URI  = str(ROOT / "mlruns")

# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if not GOLD_DIR.exists():
        print(f"ERROR: No se encuentra el Gold en {GOLD_DIR}")
        print("Ejecuta primero: python pipelines/03_silver_to_gold.py")
        sys.exit(1)

    xgb_model, baseline_model, metrics = run_training(
        gold_dir=GOLD_DIR,
        models_dir=MODELS_DIR,
        mlflow_uri=MLFLOW_URI,
    )

    print(f"\nModelo guardado en  : {MODELS_DIR / 'xgboost_churn.pkl'}")
    print(f"MLflow UI           : mlflow ui --port 5000")
    print(f"ROC-AUC XGBoost     : {metrics['xgboost']['roc_auc']:.4f}")
    print(f"ROC-AUC Baseline    : {metrics['baseline']['roc_auc']:.4f}")