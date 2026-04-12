"""
src/feature_experiments.py
===========================
Experimentos de selección de features sobre el dataset Gold.

Objetivo:
  Demostrar de forma trazable qué impacto tiene cada decisión
  de feature engineering sobre las métricas del modelo.
  Cada experimento queda registrado en MLflow con sus features y métricas.

Experimentos definidos:
  01_baseline         - Gold completo tal cual (referencia)
  02_no_redundant     - Sin is_month_to_month (colineal con contract)
  03_no_derived       - Sin ninguna feature derivada creada por nosotros
  04_top_shap_only    - Solo top 10 features por importancia SHAP
  05_no_total_charges - Sin total_charges (colineal con tenure x monthly)

Salidas:
  - Todos los runs en MLflow bajo el experimento 'churn-feature-experiments'
  - reports/feature_experiments/comparativa.csv
  - reports/feature_experiments/comparativa.png
"""

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

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

RANDOM_STATE      = 42
MLFLOW_EXPERIMENT = "churn-feature-experiments"

# Features derivadas que creamos en silver_to_gold.py
DERIVED_FEATURES = [
    "n_services",
    "is_month_to_month",
    "charges_per_tenure",
    "has_fiber",
    "is_new_customer",
    "is_loyal_customer",
]

# Top 10 features por SHAP (orden del gráfico generado en evaluate.py)
# Actualizar si el modelo ganador cambia
TOP_SHAP_FEATURES = [
    "contract",
    "charges_per_tenure",
    "is_month_to_month",
    "internet_service_Fiber optic",
    "monthly_charges",
    "payment_method_Electronic check",
    "tenure_months",
    "has_fiber",
    "paperless_billing",
    "total_charges",
]


# ---------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------

def load_gold(gold_dir: Path) -> tuple:
    X_train = pd.read_parquet(gold_dir / "X_train.parquet")
    X_test  = pd.read_parquet(gold_dir / "X_test.parquet")
    y_train = pd.read_parquet(gold_dir / "y_train.parquet").squeeze()
    y_test  = pd.read_parquet(gold_dir / "y_test.parquet").squeeze()
    logger.info(f"  Gold cargado: {len(X_train):,} train | {len(X_test):,} test")
    logger.info(f"  Features disponibles: {len(X_train.columns)}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. DEFINICIÓN DE EXPERIMENTOS
# ---------------------------------------------------------------------------

def define_experiments(X_train: pd.DataFrame) -> dict:
    """
    Define los conjuntos de features para cada experimento.
    Cada entrada es un dict con 'features' y 'description'.
    """
    all_features = X_train.columns.tolist()

    # Features sin las derivadas
    no_derived = [f for f in all_features if f not in DERIVED_FEATURES]

    # Sin is_month_to_month (colineal con contract=0)
    no_redundant = [f for f in all_features if f != "is_month_to_month"]

    # Sin total_charges (colineal con tenure_months x monthly_charges)
    no_total_charges = [f for f in all_features if f != "total_charges"]

    # Solo top SHAP que existan en el dataset
    top_shap = [f for f in TOP_SHAP_FEATURES if f in all_features]

    experiments = {
        "01_baseline": {
            "features":    all_features,
            "description": "Gold completo — punto de referencia",
            "n_features":  len(all_features),
        },
        "02_no_redundant": {
            "features":    no_redundant,
            "description": "Sin is_month_to_month (colineal con contract)",
            "n_features":  len(no_redundant),
        },
        "03_no_derived": {
            "features":    no_derived,
            "description": "Sin features derivadas (solo variables originales)",
            "n_features":  len(no_derived),
        },
        "04_top_shap_only": {
            "features":    top_shap,
            "description": "Solo top 10 features por importancia SHAP",
            "n_features":  len(top_shap),
        },
        "05_no_total_charges": {
            "features":    no_total_charges,
            "description": "Sin total_charges (colineal con tenure x monthly_charges)",
            "n_features":  len(no_total_charges),
        },
    }

    logger.info("  Experimentos definidos:")
    for name, exp in experiments.items():
        logger.info(f"    {name}: {exp['n_features']} features — {exp['description']}")

    return experiments


# ---------------------------------------------------------------------------
# 3. ENTRENAMIENTO POR EXPERIMENTO
# ---------------------------------------------------------------------------

def _get_scale_pos_weight(y_train: pd.Series) -> float:
    return (y_train == 0).sum() / (y_train == 1).sum()


def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "roc_auc":       round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
        "f1":            round(f1_score(y_true, y_pred), 4),
        "precision":     round(precision_score(y_true, y_pred), 4),
        "recall":        round(recall_score(y_true, y_pred), 4),
    }


def train_experiment(
    exp_name:   str,
    features:   list,
    X_train:    pd.DataFrame,
    X_test:     pd.DataFrame,
    y_train:    pd.Series,
    y_test:     pd.Series,
    best_model_path: Path,
) -> dict:
    """
    Entrena el modelo ganador (mismos hiperparámetros que en 04_train)
    pero con el subconjunto de features del experimento.

    Usamos los hiperparámetros del mejor modelo guardado para que
    la comparación sea justa: lo único que cambia son las features.
    """
    # Cargar hiperparámetros del mejor modelo entrenado
    artifact   = joblib.load(best_model_path)
    best_model = artifact["model"]

    # Extraer parámetros (excluir los que no son del constructor)
    params = best_model.get_params()

    # Reentrenar con el subconjunto de features
    scale_pos_weight = _get_scale_pos_weight(y_train)
    params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**params)
    model.fit(
        X_train[features], y_train,
        eval_set=[(X_test[features], y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test[features])
    y_prob = model.predict_proba(X_test[features])[:, 1]
    metrics = _compute_metrics(y_test, y_pred, y_prob)

    logger.info(
        f"  {exp_name}: ROC-AUC={metrics['roc_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | features={len(features)}"
    )

    return metrics


# ---------------------------------------------------------------------------
# 4. MLFLOW LOGGING
# ---------------------------------------------------------------------------

def log_experiment_to_mlflow(
    exp_name:    str,
    description: str,
    features:    list,
    metrics:     dict,
    mlflow_uri:  str,
) -> None:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=exp_name):
        mlflow.set_tag("description",  description)
        mlflow.set_tag("experiment_id", exp_name)

        mlflow.log_param("n_features", len(features))
        mlflow.log_param("features",   str(features))
        mlflow.log_param("description", description)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)


# ---------------------------------------------------------------------------
# 5. COMPARATIVA VISUAL Y CSV
# ---------------------------------------------------------------------------

def save_comparison(results: dict, reports_dir: Path) -> pd.DataFrame:
    """
    Guarda un CSV con la comparativa de todos los experimentos
    y genera un gráfico de barras comparativo.
    """
    rows = []
    for exp_name, data in results.items():
        row = {"experimento": exp_name, "n_features": data["n_features"]}
        row.update(data["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("experimento")
    df = df.sort_values("roc_auc", ascending=False)

    # Guardar CSV
    csv_path = reports_dir / "comparativa.csv"
    df.to_csv(csv_path)
    logger.info(f"  CSV guardado: {csv_path}")

    # Gráfico comparativo
    metric_keys = ["roc_auc", "avg_precision", "f1", "recall"]
    fig, axes   = plt.subplots(1, len(metric_keys), figsize=(16, 5))

    colors_base = ["#378ADD", "#E24B4A", "#1D9E75", "#EF9F27", "#9FE1CB"]
    exp_names   = df.index.tolist()
    colors      = colors_base[:len(exp_names)]

    for ax, metric in zip(axes, metric_keys):
        values = df[metric].values
        bars   = ax.barh(exp_names, values, color=colors, alpha=0.85)

        # Valor encima de cada barra
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8,
            )

        # Línea del baseline para referencia
        baseline_val = df.loc["01_baseline", metric] if "01_baseline" in df.index else None
        if baseline_val is not None:
            ax.axvline(x=baseline_val, color="gray", linestyle="--",
                       linewidth=1, alpha=0.7, label="baseline")

        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("Score")
        max_val = max(values)
        ax.set_xlim([max(0, min(values) - 0.05), min(1, max_val + 0.05)])

    plt.suptitle(
        "Comparativa de experimentos de features",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    fig_path = reports_dir / "comparativa.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Gráfico guardado: {fig_path}")

    return df


def print_comparison_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 75)
    print("COMPARATIVA DE EXPERIMENTOS DE FEATURES")
    print("=" * 75)

    metric_keys = ["n_features", "roc_auc", "avg_precision", "f1", "recall"]
    header = f"  {'Experimento':30}" + "".join(f"{m:>13}" for m in metric_keys)
    print(header)
    print("  " + "-" * 73)

    baseline_auc = df.loc["01_baseline", "roc_auc"] if "01_baseline" in df.index else None

    for exp_name, row in df.iterrows():
        line = f"  {exp_name:30}"
        line += f"{int(row['n_features']):>13}"
        for m in metric_keys[1:]:
            val  = row[m]
            line += f"{val:>13.4f}"
        # Indicar diferencia vs baseline
        if baseline_auc and exp_name != "01_baseline":
            diff = row["roc_auc"] - baseline_auc
            sign = "+" if diff >= 0 else ""
            line += f"  ({sign}{diff:.4f} vs baseline)"
        print(line)

    print("\n  Interpretación:")
    best     = df["roc_auc"].idxmax()
    worst    = df["roc_auc"].idxmin()
    print(f"  · Mejor experimento por ROC-AUC : {best}")
    print(f"  · Peor experimento por ROC-AUC  : {worst}")

    if "01_baseline" in df.index and "03_no_derived" in df.index:
        diff_derived = df.loc["01_baseline", "roc_auc"] - df.loc["03_no_derived", "roc_auc"]
        print(f"  · Impacto de las features derivadas: +{diff_derived:.4f} ROC-AUC vs sin ellas")

    if "01_baseline" in df.index and "02_no_redundant" in df.index:
        diff_redund = df.loc["01_baseline", "roc_auc"] - df.loc["02_no_redundant", "roc_auc"]
        direction   = "pierde" if diff_redund > 0 else "gana"
        print(f"  · Eliminar is_month_to_month: {direction} {abs(diff_redund):.4f} ROC-AUC")

    print("=" * 75)


# ---------------------------------------------------------------------------
# FUNCIÓN ORQUESTRADORA
# ---------------------------------------------------------------------------

def run_feature_experiments(
    gold_dir:        str | Path,
    models_dir:      str | Path,
    reports_dir:     str | Path,
    mlflow_uri:      str,
) -> pd.DataFrame:
    """
    Ejecuta todos los experimentos de features y devuelve la comparativa.
    """
    gold_dir    = Path(gold_dir)
    models_dir  = Path(models_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = models_dir / "best_model.pkl"

    logger.info("=" * 60)
    logger.info("PIPELINE: Experimentos de features")
    logger.info("=" * 60)

    logger.info("[1/4] Cargando Gold...")
    X_train, X_test, y_train, y_test = load_gold(gold_dir)

    logger.info("[2/4] Definiendo experimentos...")
    experiments = define_experiments(X_train)

    logger.info("[3/4] Entrenando y evaluando cada experimento...")
    results = {}
    for exp_name, exp_config in experiments.items():
        logger.info(f"  → {exp_name} ({exp_config['n_features']} features)...")

        metrics = train_experiment(
            exp_name        = exp_name,
            features        = exp_config["features"],
            X_train         = X_train,
            X_test          = X_test,
            y_train         = y_train,
            y_test          = y_test,
            best_model_path = best_model_path,
        )

        log_experiment_to_mlflow(
            exp_name    = exp_name,
            description = exp_config["description"],
            features    = exp_config["features"],
            metrics     = metrics,
            mlflow_uri  = mlflow_uri,
        )

        results[exp_name] = {
            "metrics":     metrics,
            "n_features":  exp_config["n_features"],
            "description": exp_config["description"],
            "features":    exp_config["features"],
        }

    logger.info("[4/4] Generando comparativa...")
    df_comparison = save_comparison(results, reports_dir)
    print_comparison_table(df_comparison)

    logger.info("=" * 60)
    logger.info("EXPERIMENTOS COMPLETADOS")
    logger.info(f"  CSV y gráfico en: {reports_dir}")
    logger.info("=" * 60)

    return df_comparison


# ---------------------------------------------------------------------------
# EJECUCIÓN DIRECTA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parent.parent

    gold_dir    = ROOT / "data" / "gold"
    models_dir  = ROOT / "models"
    reports_dir = ROOT / "reports" / "feature_experiments"
    mlflow_uri  = str(ROOT / "mlruns")

    if not (models_dir / "best_model.pkl").exists():
        logger.error("No se encuentra best_model.pkl. Ejecuta primero: python pipelines/04_train.py")
        sys.exit(1)

    run_feature_experiments(gold_dir, models_dir, reports_dir, mlflow_uri)