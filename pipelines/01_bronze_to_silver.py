"""
pipelines/01_bronze_to_silver.py
=================================
Ejecuta el pipeline de transformación Bronze → Silver.
Uso: python pipelines/01_bronze_to_silver.py
"""

import sys
from pathlib import Path

# Añadir src al path para poder importar el módulo
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bronze_to_silver import run_bronze_to_silver

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

BRONZE_PATH = ROOT / "data" / "bronze" / "raw_data.csv"
SILVER_PATH = ROOT / "data" / "silver" / "customers_clean.parquet"

# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if not BRONZE_PATH.exists():
        print(f"ERROR: No se encuentra el fichero Bronze en {BRONZE_PATH}")
        sys.exit(1)

    df_silver, quality_report = run_bronze_to_silver(
        bronze_path=BRONZE_PATH,
        silver_path=SILVER_PATH,
    )

    print("\nResumen final:")
    print(f"  Registros en Silver : {len(df_silver):,}")
    print(f"  Churn rate          : {(df_silver['churn'] == 'Yes').mean():.1%}")
    print(f"  Guardado en         : {SILVER_PATH}")