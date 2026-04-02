"""
pipelines/03_silver_to_gold.py
================================
Ejecuta el feature engineering Silver → Gold.
Uso: python pipelines/03_silver_to_gold.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from silver_to_gold import run_silver_to_gold

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

SILVER_PATH = ROOT / "data" / "silver" / "customers_clean.parquet"
GOLD_DIR    = ROOT / "data" / "gold"

# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if not SILVER_PATH.exists():
        print(f"ERROR: No se encuentra el Silver en {SILVER_PATH}")
        print("Ejecuta primero: python pipelines/01_bronze_to_silver.py")
        sys.exit(1)

    X_train, X_test, y_train, y_test = run_silver_to_gold(
        silver_path=SILVER_PATH,
        gold_dir=GOLD_DIR,
    )

    print(f"\nResumen Gold:")
    print(f"  Features totales : {len(X_train.columns)}")
    print(f"  Train size       : {len(X_train):,}")
    print(f"  Test size        : {len(X_test):,}")
    print(f"  Churn rate train : {y_train.mean():.1%}")
    print(f"  Churn rate test  : {y_test.mean():.1%}")
    print(f"  Guardado en      : {GOLD_DIR}")