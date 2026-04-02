"""
src/silver_to_gold.py
=====================
Feature engineering sobre la capa Silver → Gold.
El dataset Gold es el input directo del modelo de churn.

Qué ocurre aquí (NO en Silver, NO en modelado):
  - Encoding de variables categóricas
  - Creación de features derivadas
  - Train/test split estratificado
  - Guardado de X_train, X_test, y_train, y_test en Gold
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Mapeo ordinal para contract (tiene orden natural claro)
CONTRACT_ORDINAL = {
    "Month-to-month": 0,
    "One year":       1,
    "Two year":       2,
}

# Columnas de servicios para contar cuántos tiene el cliente
SERVICE_COLS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]

# Columnas one-hot (sin orden natural, cardinalidad baja-media)
ONE_HOT_COLS = ["internet_service", "payment_method"]

# Columnas binarias Yes/No → 0/1
BINARY_COLS = [
    "partner", "dependents", "phone_service", "multiple_lines",
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
    "paperless_billing",
]


# ---------------------------------------------------------------------------
# 1. CARGA DESDE SILVER
# ---------------------------------------------------------------------------

def load_silver(silver_path: str | Path) -> pd.DataFrame:
    path = Path(silver_path)
    logger.info(f"Cargando Silver desde: {path}")
    df = pd.read_parquet(path)
    logger.info(f"  {len(df):,} registros | {df.shape[1]} columnas")
    return df


# ---------------------------------------------------------------------------
# 2. LIMPIEZA PREVIA AL ENGINEERING
# ---------------------------------------------------------------------------

def drop_metadata_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas de metadatos de pipeline (_ingestion_timestamp, etc.)
    y customer_id que no son features del modelo.
    """
    meta_cols = [c for c in df.columns if c.startswith("_")]
    drop_cols = meta_cols + ["customer_id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    logger.info(f"  Eliminadas columnas no-feature: {drop_cols}")
    return df


# ---------------------------------------------------------------------------
# 3. ENCODING
# ---------------------------------------------------------------------------

def encode_binary_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yes → 1, No → 0 en todas las columnas binarias.
    gender: Female → 0, Male → 1.
    """
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1})

    logger.info(f"  Encoding binario aplicado a {len(BINARY_COLS) + 1} columnas.")
    return df


def encode_contract_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contract tiene orden natural: Month-to-month < One year < Two year.
    Un cliente con contrato más largo tiene menor riesgo → ordinal tiene sentido.
    """
    df["contract"] = df["contract"].map(CONTRACT_ORDINAL)
    logger.info(f"  Encoding ordinal aplicado a 'contract': {CONTRACT_ORDINAL}")
    return df


def encode_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encoding para internet_service y payment_method.
    drop_first=True para evitar multicolinealidad perfecta.
    """
    for col in ONE_HOT_COLS:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        logger.info(f"  One-hot '{col}': {list(dummies.columns)}")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Churn: Yes → 1, No → 0.
    """
    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})
    logger.info("  Target 'churn' codificado: Yes=1, No=0.")
    return df


# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features derivadas que aportan señal adicional al modelo.

    Justificación de cada feature:
    - n_services:          Clientes con más servicios tienen más engagement → menor churn
    - is_month_to_month:   El contrato mensual es el mayor predictor de churn en el EDA
    - charges_per_tenure:  Coste relativo al tiempo como cliente (¿está pagando mucho por poco tiempo?)
    - has_fiber:           Fiber optic tiene el churn rate más alto de internet_service
    - is_new_customer:     Clientes con tenure < 12 meses tienen churn elevado
    - is_loyal_customer:   Clientes con tenure > 48 meses raramente hacen churn
    """

    # Número de servicios contratados
    # En este punto BINARY_COLS ya están codificados como 0/1
    service_cols_present = [c for c in SERVICE_COLS if c in df.columns]
    df["n_services"] = df[service_cols_present].sum(axis=1)
    logger.info(f"  Feature 'n_services': suma de {len(service_cols_present)} servicios.")

    # Flag: contrato mensual (el más predictivo según EDA)
    if "contract" in df.columns:
        df["is_month_to_month"] = (df["contract"] == 0).astype(int)
        logger.info("  Feature 'is_month_to_month': contract == 0.")

    # Ratio coste mensual por mes de antigüedad (evitar división por 0)
    if "monthly_charges" in df.columns and "tenure_months" in df.columns:
        df["charges_per_tenure"] = (
            df["monthly_charges"] / (df["tenure_months"] + 1)
        ).round(4)
        logger.info("  Feature 'charges_per_tenure': monthly_charges / (tenure_months + 1).")

    # Flag: tiene fibra óptica (mayor churn rate que DSL o sin internet)
    fiber_col = "internet_service_Fiber optic"
    if fiber_col in df.columns:
        df["has_fiber"] = df[fiber_col].astype(int)
        logger.info("  Feature 'has_fiber': desde internet_service one-hot.")

    # Segmentos de tenure
    if "tenure_months" in df.columns:
        df["is_new_customer"]   = (df["tenure_months"] < 12).astype(int)
        df["is_loyal_customer"] = (df["tenure_months"] > 48).astype(int)
        logger.info("  Features 'is_new_customer' y 'is_loyal_customer' creadas.")

    return df


# ---------------------------------------------------------------------------
# 5. VALIDACIÓN DEL GOLD
# ---------------------------------------------------------------------------

def validate_gold(df: pd.DataFrame) -> None:
    """
    Checks básicos antes de guardar el Gold:
    - Sin nulos
    - Sin columnas object (todo debe estar codificado)
    - Target presente y binario
    """
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        null_detail = df.isnull().sum()[df.isnull().sum() > 0]
        raise ValueError(f"Gold tiene {nulls} nulos:\n{null_detail}")

    object_cols = df.select_dtypes(include="object").columns.tolist()
    if object_cols:
        raise ValueError(f"Columnas sin codificar en Gold: {object_cols}")

    if "churn" not in df.columns:
        raise ValueError("La columna target 'churn' no está en el dataset.")

    target_vals = set(df["churn"].unique())
    if not target_vals.issubset({0, 1}):
        raise ValueError(f"Target 'churn' tiene valores inesperados: {target_vals}")

    logger.info(f"  Validación Gold: OK — {len(df):,} filas, {df.shape[1]} columnas, sin nulos.")


# ---------------------------------------------------------------------------
# 6. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------

def split_dataset(df: pd.DataFrame) -> tuple:
    """
    Split estratificado para preservar la proporción de churn en ambos sets.
    80% train / 20% test.
    Devuelve X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(f"  Train: {len(X_train):,} filas | Churn rate: {y_train.mean():.1%}")
    logger.info(f"  Test:  {len(X_test):,} filas  | Churn rate: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 7. GUARDADO EN GOLD
# ---------------------------------------------------------------------------

def save_gold(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    gold_dir: str | Path,
) -> None:
    """
    Guarda los 4 splits en Gold como Parquet.
    También guarda el dataset completo (gold_full) para referencia.
    """
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(gold_dir / "X_train.parquet", index=False)
    X_test.to_parquet( gold_dir / "X_test.parquet",  index=False)
    y_train.to_frame().to_parquet(gold_dir / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(  gold_dir / "y_test.parquet",  index=False)

    logger.info(f"  Gold guardado en: {gold_dir}")
    logger.info(f"  Ficheros: X_train, X_test, y_train, y_test (.parquet)")
    logger.info(f"  Features del modelo: {list(X_train.columns)}")


# ---------------------------------------------------------------------------
# FUNCIÓN ORQUESTRADORA
# ---------------------------------------------------------------------------

def run_silver_to_gold(
    silver_path: str | Path,
    gold_dir:    str | Path,
) -> tuple:
    """
    Ejecuta el pipeline completo Silver → Gold.

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("=" * 60)
    logger.info("PIPELINE: Silver → Gold | Feature Engineering")
    logger.info("=" * 60)

    logger.info("[1/7] Cargando Silver...")
    df = load_silver(silver_path)

    logger.info("[2/7] Eliminando columnas no-feature...")
    df = drop_metadata_cols(df)

    logger.info("[3/7] Encoding de variables...")
    df = encode_target(df)
    df = encode_binary_cols(df)
    df = encode_contract_ordinal(df)
    df = encode_one_hot(df)

    logger.info("[4/7] Creando features derivadas...")
    df = create_derived_features(df)

    logger.info("[5/7] Validando Gold...")
    validate_gold(df)

    logger.info("[6/7] Train/test split estratificado...")
    X_train, X_test, y_train, y_test = split_dataset(df)

    logger.info("[7/7] Guardando Gold...")
    save_gold(X_train, X_test, y_train, y_test, gold_dir)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# EJECUCIÓN DIRECTA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parent.parent

    silver_path = ROOT / "data" / "silver" / "customers_clean.parquet"
    gold_dir    = ROOT / "data" / "gold"

    if not silver_path.exists():
        logger.error(f"No se encuentra el Silver: {silver_path}")
        sys.exit(1)

    X_train, X_test, y_train, y_test = run_silver_to_gold(silver_path, gold_dir)

    print(f"\nFeatures del modelo ({len(X_train.columns)}):")
    for col in sorted(X_train.columns):
        print(f"  {col}")