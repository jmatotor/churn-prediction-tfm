"""
pipelines/02_eda_silver.py
===========================
Ejecuta el análisis exploratorio sobre la capa Silver.
Los gráficos se guardan en reports/eda/.

Uso: python pipelines/02_eda_silver.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eda_silver import run_eda

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

SILVER_PATH = ROOT / "data" / "silver" / "customers_clean.parquet"
REPORTS_DIR = ROOT / "reports" / "eda"

# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if not SILVER_PATH.exists():
        print(f"ERROR: No se encuentra el Silver en {SILVER_PATH}")
        print("Ejecuta primero: python pipelines/01_bronze_to_silver.py")
        sys.exit(1)

    run_eda(
        silver_path=SILVER_PATH,
        reports_dir=REPORTS_DIR,
    )

    print(f"\nGráficos disponibles en: {REPORTS_DIR}")