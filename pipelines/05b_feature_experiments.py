"""
pipelines/05b_feature_experiments.py
======================================
Ejecuta los experimentos de selección de features.

Compara 5 configuraciones distintas del dataset Gold
manteniendo el mismo modelo y mismos hiperparámetros.
Todo queda registrado en MLflow bajo 'churn-feature-experiments'.

Uso: python pipelines/05b_feature_experiments.py

Salidas:
  reports/feature_experiments/comparativa.csv
  reports/feature_experiments/comparativa.png
  MLflow: experimento 'churn-feature-experiments' con 5 runs
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from feature_experiments import run_feature_experiments

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

GOLD_DIR    = ROOT / "data" / "gold"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "feature_experiments"
MLFLOW_URI  = (ROOT / "mlruns").resolve().as_uri()

# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if not (MODELS_DIR / "best_model.pkl").exists():
        print("ERROR: No se encuentra best_model.pkl")
        print("Ejecuta primero: python pipelines/04_train.py")
        sys.exit(1)

    if not GOLD_DIR.exists():
        print("ERROR: No se encuentra el Gold")
        print("Ejecuta primero: python pipelines/03_silver_to_gold.py")
        sys.exit(1)

    df = run_feature_experiments(
        gold_dir    = GOLD_DIR,
        models_dir  = MODELS_DIR,
        reports_dir = REPORTS_DIR,
        mlflow_uri  = MLFLOW_URI,
    )

    print(f"\nExperimentos completados: {len(df)}")
    print(f"Mejor configuración     : {df['roc_auc'].idxmax()}")
    print(f"Resultados en           : {REPORTS_DIR}")
    print(f"MLflow UI               : mlflow ui --port 5000")