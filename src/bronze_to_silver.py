"""
src/bronze_to_silver.py
=======================
Lógica de transformación y validación para el paso Bronze → Silver.
Diseñado para ser importado desde el notebook del pipeline o ejecutado directamente.

Principio ELT aplicado:
  - Bronze: datos raw, sin tocar (fuente de verdad)
  - Silver: datos limpios, tipados y validados, listos para feature engineering
  - Gold:   features modeladas (siguiente capa, no se toca aquí)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONSTANTES DEL ESQUEMA
# ---------------------------------------------------------------------------

# Mapeo de nombres originales → snake_case estandarizado
COLUMN_RENAME_MAP = {
    "customerID":       "customer_id",
    "gender":           "gender",
    "SeniorCitizen":    "senior_citizen",
    "Partner":          "partner",
    "Dependents":       "dependents",
    "tenure":           "tenure_months",
    "PhoneService":     "phone_service",
    "MultipleLines":    "multiple_lines",
    "InternetService":  "internet_service",
    "OnlineSecurity":   "online_security",
    "OnlineBackup":     "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport":      "tech_support",
    "StreamingTV":      "streaming_tv",
    "StreamingMovies":  "streaming_movies",
    "Contract":         "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod":    "payment_method",
    "MonthlyCharges":   "monthly_charges",
    "TotalCharges":     "total_charges",
    "Churn":            "churn",
}

# Columnas binarias Yes/No que serán estandarizadas (no codificadas aquí, eso va en Gold)
BINARY_YES_NO_COLS = [
    "partner", "dependents", "phone_service", "paperless_billing", "churn",
]

# Columnas de servicios que pueden tener "No internet service" o "No phone service"
# → se normalizan a "No" para consistencia
SERVICE_COLS = [
    "multiple_lines", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
]

# Tipos esperados en Silver (antes del feature engineering)
EXPECTED_DTYPES = {
    "customer_id":      "object",
    "gender":           "object",
    "senior_citizen":   "int64",
    "partner":          "object",
    "dependents":       "object",
    "tenure_months":    "int64",
    "phone_service":    "object",
    "multiple_lines":   "object",
    "internet_service": "object",
    "online_security":  "object",
    "online_backup":    "object",
    "device_protection":"object",
    "tech_support":     "object",
    "streaming_tv":     "object",
    "streaming_movies": "object",
    "contract":         "object",
    "paperless_billing":"object",
    "payment_method":   "object",
    "monthly_charges":  "float64",
    "total_charges":    "float64",
    "churn":            "object",
}


# ---------------------------------------------------------------------------
# 1. CARGA DESDE BRONZE
# ---------------------------------------------------------------------------

def load_bronze(bronze_path: str | Path) -> pd.DataFrame:
    """
    Carga el CSV raw desde la capa Bronze sin ninguna transformación.
    Se registra el tamaño y el esquema para trazabilidad.
    """
    path = Path(bronze_path)
    logger.info(f"Cargando Bronze desde: {path}")

    df = pd.read_csv(path)

    logger.info(f"  Filas cargadas:   {len(df):,}")
    logger.info(f"  Columnas:         {df.shape[1]}")
    logger.info(f"  Memoria aprox:    {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    return df


# ---------------------------------------------------------------------------
# 2. RENOMBRADO Y ESTANDARIZACIÓN DE ESQUEMA
# ---------------------------------------------------------------------------

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas al estándar snake_case del proyecto.
    Falla explícitamente si el CSV no contiene las columnas esperadas.
    """
    missing = set(COLUMN_RENAME_MAP.keys()) - set(df.columns)
    if missing:
        raise ValueError(
            f"Columnas no encontradas en el CSV de Bronze: {missing}\n"
            "Verifica que el fichero es el dataset de Telco Customer Churn."
        )

    df = df.rename(columns=COLUMN_RENAME_MAP)
    logger.info("  Columnas renombradas a snake_case.")
    return df


# ---------------------------------------------------------------------------
# 3. LIMPIEZA DE TIPOS Y VALORES
# ---------------------------------------------------------------------------

def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges viene como string con espacios vacíos para clientes
    con tenure == 0 (recién dados de alta, sin primer cobro).
    
    Decisión de negocio: imputar con monthly_charges (primer mes prorrateado)
    en lugar de 0, que es más representativo de la realidad.
    """
    n_before = df["total_charges"].dtype

    # Reemplazar espacios vacíos con NaN y convertir a float
    df["total_charges"] = pd.to_numeric(df["total_charges"].str.strip(), errors="coerce")

    n_nulls = df["total_charges"].isna().sum()

    if n_nulls > 0:
        logger.info(
            f"  total_charges: {n_nulls} valores nulos detectados "
            f"(clientes con tenure=0). Imputando con monthly_charges."
        )
        # Impute: si total_charges es NaN y tenure == 0, usar monthly_charges
        mask = df["total_charges"].isna() & (df["tenure_months"] == 0)
        df.loc[mask, "total_charges"] = df.loc[mask, "monthly_charges"]

        # Si quedaran NaN por otra razón, imputar con mediana (safety net)
        remaining_nulls = df["total_charges"].isna().sum()
        if remaining_nulls > 0:
            median_val = df["total_charges"].median()
            df["total_charges"].fillna(median_val, inplace=True)
            logger.warning(
                f"  {remaining_nulls} NaN en total_charges no relacionados con tenure=0. "
                f"Imputados con mediana ({median_val:.2f})."
            )

    logger.info(f"  total_charges: convertido a float64.")
    return df


def normalize_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Columnas de servicios tienen tres valores posibles:
      'Yes', 'No', 'No internet service' / 'No phone service'
    
    'No internet service' y 'No phone service' son equivalentes a 'No'
    en el contexto del servicio específico (el cliente simplemente no lo tiene).
    Se unifica a 'No' para simplificar el feature engineering posterior.
    
    Nota: la información de si tiene internet o teléfono ya está capturada
    en internet_service y phone_service respectivamente.
    """
    replacements = {
        "No internet service": "No",
        "No phone service": "No",
    }

    for col in SERVICE_COLS:
        if col in df.columns:
            before = df[col].value_counts().to_dict()
            df[col] = df[col].replace(replacements)
            logger.info(f"  {col}: normalizado (antes: {before})")

    return df


def standardize_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que las columnas binarias Yes/No tienen exactamente esos valores,
    sin espacios ni capitalización distinta.
    En Silver mantenemos Yes/No como strings (la codificación va en Gold).
    """
    for col in BINARY_YES_NO_COLS:
        if col in df.columns:
            df[col] = df[col].str.strip().str.capitalize()

    logger.info(f"  Columnas binarias estandarizadas: {BINARY_YES_NO_COLS}")
    return df


def fix_senior_citizen(df: pd.DataFrame) -> pd.DataFrame:
    """
    senior_citizen viene como 0/1 (int) mientras el resto de binarias son Yes/No.
    Para Silver lo dejamos como int (0/1) ya que es el tipo más limpio
    para esta variable. Documentado explícitamente para el feature engineering.
    """
    # Asegurar que es int, no float
    df["senior_citizen"] = df["senior_citizen"].astype("int64")
    logger.info("  senior_citizen: confirmado como int64 (0/1).")
    return df


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuerza los tipos de datos del esquema Silver definido en EXPECTED_DTYPES.
    Lanza warning si no puede convertir alguna columna.
    """
    for col, expected_dtype in EXPECTED_DTYPES.items():
        if col not in df.columns:
            continue
        try:
            df[col] = df[col].astype(expected_dtype)
        except (ValueError, TypeError) as e:
            logger.warning(f"  No se pudo convertir {col} a {expected_dtype}: {e}")

    logger.info("  Tipos de datos del esquema Silver aplicados.")
    return df


# ---------------------------------------------------------------------------
# 4. DEDUPLICACIÓN Y VALIDACIÓN DE CLAVE PRIMARIA
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas duplicadas. customer_id debería ser único.
    Se loguean los casos detectados para trazabilidad.
    """
    n_before = len(df)

    # Duplicados exactos (todas las columnas)
    exact_dups = df.duplicated().sum()
    if exact_dups > 0:
        df = df.drop_duplicates()
        logger.warning(f"  Eliminados {exact_dups} duplicados exactos.")

    # Duplicados por customer_id (mismo cliente, datos distintos)
    id_dups = df.duplicated(subset=["customer_id"]).sum()
    if id_dups > 0:
        logger.warning(
            f"  {id_dups} customer_id duplicados detectados. "
            "Conservando primera aparición."
        )
        df = df.drop_duplicates(subset=["customer_id"], keep="first")

    n_after = len(df)
    if n_before != n_after:
        logger.info(f"  Filas tras deduplicación: {n_after:,} (eliminadas: {n_before - n_after})")

    return df


def validate_primary_key(df: pd.DataFrame) -> None:
    """
    Validación de integridad: customer_id debe ser único y no nulo.
    Lanza excepción si no se cumple (fail fast, no silenciar errores de datos).
    """
    nulls = df["customer_id"].isna().sum()
    if nulls > 0:
        raise ValueError(f"customer_id tiene {nulls} valores nulos. Pipeline abortado.")

    dupes = df["customer_id"].duplicated().sum()
    if dupes > 0:
        raise ValueError(f"customer_id tiene {dupes} duplicados tras limpieza. Pipeline abortado.")

    logger.info(f"  customer_id validado: {len(df):,} registros únicos y no nulos.")


# ---------------------------------------------------------------------------
# 5. AÑADIR METADATOS DE PIPELINE
# ---------------------------------------------------------------------------

def add_pipeline_metadata(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """
    Añade columnas de metadatos para trazabilidad del pipeline.
    Permite saber de dónde y cuándo vino cada registro en Silver.
    """
    df["_ingestion_timestamp"] = datetime.utcnow().isoformat()
    df["_source_file"] = Path(source_file).name
    df["_pipeline_version"] = "1.0.0"
    df["_layer"] = "silver"

    logger.info("  Metadatos de pipeline añadidos.")
    return df


# ---------------------------------------------------------------------------
# 6. VALIDACIÓN FINAL (DATA QUALITY CHECKS)
# ---------------------------------------------------------------------------

def run_quality_checks(df: pd.DataFrame) -> dict:
    """
    Ejecuta checks de calidad sobre el dataset Silver.
    Devuelve un dict con el resumen para logging/reporting.
    No bloquea el pipeline, pero emite warnings claros.
    """
    report = {}

    # Check 1: Nulos por columna
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    report["columns_with_nulls"] = null_cols.to_dict()
    if not null_cols.empty:
        logger.warning(f"  QUALITY CHECK - Columnas con nulos:\n{null_cols}")
    else:
        logger.info("  QUALITY CHECK - Sin nulos en columnas de negocio. OK")

    # Check 2: Balance de la variable target
    churn_dist = df["churn"].value_counts(normalize=True)
    report["churn_distribution"] = churn_dist.to_dict()
    churn_rate = churn_dist.get("Yes", 0)
    logger.info(f"  QUALITY CHECK - Tasa de churn: {churn_rate:.1%} (esperado ~26%)")
    if not (0.20 <= churn_rate <= 0.35):
        logger.warning(f"  Tasa de churn fuera de rango esperado: {churn_rate:.1%}")

    # Check 3: Rango de variables numéricas
    for col, (min_exp, max_exp) in {
        "tenure_months":    (0, 120),
        "monthly_charges":  (0, 200),
        "total_charges":    (0, 15000),
    }.items():
        actual_min = df[col].min()
        actual_max = df[col].max()
        report[f"{col}_range"] = (actual_min, actual_max)
        if actual_min < min_exp or actual_max > max_exp:
            logger.warning(
                f"  QUALITY CHECK - {col} fuera de rango esperado "
                f"[{min_exp}, {max_exp}]: actual [{actual_min:.2f}, {actual_max:.2f}]"
            )
        else:
            logger.info(f"  QUALITY CHECK - {col}: [{actual_min:.2f}, {actual_max:.2f}] OK")

    # Check 4: Valores únicos en columnas categóricas
    for col in ["gender", "contract", "internet_service", "payment_method"]:
        uniques = df[col].unique().tolist()
        report[f"{col}_unique_values"] = uniques
        logger.info(f"  QUALITY CHECK - {col}: {uniques}")

    # Check 5: Número mínimo de registros
    n = len(df)
    report["total_records"] = n
    if n < 6000:
        logger.warning(f"  QUALITY CHECK - Solo {n:,} registros. Esperado >6000.")
    else:
        logger.info(f"  QUALITY CHECK - Total registros: {n:,}. OK")

    return report


# ---------------------------------------------------------------------------
# 7. GUARDADO EN SILVER
# ---------------------------------------------------------------------------

def save_silver(df: pd.DataFrame, silver_path: str | Path) -> None:
    """
    Guarda el dataset limpio en Silver como Parquet.
    
    Por qué Parquet y no CSV:
      - Tipado nativo (los dtypes se preservan)
      - Compresión automática (~3-5x más pequeño)
      - Lectura columnar (más rápido para ML)
      - Compatible con Delta Lake, Spark, Pandas, PyArrow, etc.
    """
    path = Path(silver_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Separar columnas de metadatos para guardarlas al final
    meta_cols = [c for c in df.columns if c.startswith("_")]
    biz_cols  = [c for c in df.columns if not c.startswith("_")]

    df = df[biz_cols + meta_cols]

    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")

    size_kb = path.stat().st_size / 1024
    logger.info(f"  Silver guardado en: {path}")
    logger.info(f"  Tamaño en disco: {size_kb:.1f} KB")
    logger.info(f"  Filas: {len(df):,} | Columnas: {df.shape[1]}")


# ---------------------------------------------------------------------------
# 8. FUNCIÓN ORQUESTRADORA PRINCIPAL
# ---------------------------------------------------------------------------

def run_bronze_to_silver(
    bronze_path: str | Path,
    silver_path: str | Path,
) -> Tuple[pd.DataFrame, dict]:
    """
    Ejecuta el pipeline completo Bronze → Silver.
    
    Args:
        bronze_path: Ruta al CSV raw en la capa Bronze.
        silver_path: Ruta de destino para el Parquet en Silver.
    
    Returns:
        Tuple con (DataFrame Silver, dict con quality report)
    """
    logger.info("=" * 60)
    logger.info("PIPELINE: Bronze → Silver | Telco Customer Churn")
    logger.info("=" * 60)

    # 1. Carga
    logger.info("[1/7] Cargando datos desde Bronze...")
    df = load_bronze(bronze_path)

    # 2. Esquema
    logger.info("[2/7] Estandarizando esquema...")
    df = rename_columns(df)

    # 3. Limpieza
    logger.info("[3/7] Limpiando y corrigiendo valores...")
    df = fix_total_charges(df)
    df = normalize_service_columns(df)
    df = standardize_binary_columns(df)
    df = fix_senior_citizen(df)
    df = enforce_dtypes(df)

    # 4. Deduplicación
    logger.info("[4/7] Eliminando duplicados...")
    df = remove_duplicates(df)
    validate_primary_key(df)

    # 5. Metadatos
    logger.info("[5/7] Añadiendo metadatos de pipeline...")
    df = add_pipeline_metadata(df, source_file=str(bronze_path))

    # 6. Quality checks
    logger.info("[6/7] Ejecutando quality checks...")
    quality_report = run_quality_checks(df)

    # 7. Guardado
    logger.info("[7/7] Guardando Silver...")
    save_silver(df, silver_path)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)

    return df, quality_report


# ---------------------------------------------------------------------------
# EJECUCIÓN DIRECTA (sin notebook)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parent.parent

    bronze_path = ROOT / "data" / "bronze" / "raw_data.csv"
    silver_path = ROOT / "data" / "silver" / "customers_clean.parquet"

    if not bronze_path.exists():
        logger.error(f"No se encuentra el fichero Bronze: {bronze_path}")
        sys.exit(1)

    df_silver, report = run_bronze_to_silver(bronze_path, silver_path)
    print("\nQuality Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")