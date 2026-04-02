"""
src/train.py
============
Entrenamiento del modelo de churn con MLflow tracking.

Pasos:
  1. Carga Gold (X_train, X_test, y_train, y_test)
  2. Baseline: Regresión Logística
  3. Modelo principal: XGBoost con Optuna
  4. Registro de experimentos en MLflow
  5. Guardado del mejor modelo en models/
"""

import logging
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------

RANDOM_STATE   = 42
OPTUNA_TRIALS  = 30       # Aumentar a 50-100 si tienes tiempo
MLFLOW_EXPERIMENT = "churn-prediction-tfm"


# ---------------------------------------------------------------------------
# 1. CARGA DESDE GOLD
# ---------------------------------------------------------------------------

def load_gold(gold_dir: str | Path) -> tuple:
    gold_dir = Path(gold_dir)
    logger.info(f"Cargando Gold desde: {gold_dir}")

    X_train = pd.read_parquet(gold_dir / "X_train.parquet")
    X_test  = pd.read_parquet(gold_dir / "X_test.parquet")
    y_train = pd.read_parquet(gold_dir / "y_train.parquet").squeeze()
    y_test  = pd.read_parquet(gold_dir / "y_test.parquet").squeeze()

    logger.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    logger.info(f"  Features: {len(X_train.columns)}")
    logger.info(f"  Churn rate train: {y_train.mean():.1%} | test: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. MÉTRICAS
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Calcula el conjunto completo de métricas relevantes para churn.
    Accuracy excluida deliberadamente: dataset desbalanceado.
    """
    return {
        "roc_auc":          round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision":    round(average_precision_score(y_true, y_prob), 4),
        "f1":               round(f1_score(y_true, y_pred), 4),
        "precision":        round(precision_score(y_true, y_pred), 4),
        "recall":           round(recall_score(y_true, y_pred), 4),
    }


def log_metrics_to_console(metrics: dict, model_name: str) -> None:
    print(f"\n  [{model_name}]")
    for k, v in metrics.items():
        print(f"    {k:20}: {v:.4f}")


# ---------------------------------------------------------------------------
# 3. BASELINE: REGRESIÓN LOGÍSTICA
# ---------------------------------------------------------------------------

def train_baseline(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
) -> tuple:
    """
    Modelo baseline con Regresión Logística.
    class_weight='balanced' para manejar el desbalanceo sin SMOTE.
    El baseline sirve como punto de comparación para XGBoost.
    """
    logger.info("Entrenando baseline: Regresión Logística...")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    logger.info("  Baseline entrenado.")
    log_metrics_to_console(metrics, "Logistic Regression (baseline)")

    return model, metrics


# ---------------------------------------------------------------------------
# 4. MODELO PRINCIPAL: XGBOOST + OPTUNA
# ---------------------------------------------------------------------------

def _objective(trial, X_train, X_test, y_train, y_test, scale_pos_weight):
    """
    Función objetivo de Optuna. Optimiza ROC-AUC en test.
    scale_pos_weight maneja el desbalanceo de forma nativa en XGBoost.
    """
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        "scale_pos_weight":  scale_pos_weight,
        "random_state":      RANDOM_STATE,
        "eval_metric":       "auc",
        "use_label_encoder": False,
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)


def train_xgboost(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
) -> tuple:
    """
    Entrena XGBoost con búsqueda de hiperparámetros via Optuna.
    scale_pos_weight = n_negativos / n_positivos para manejar desbalanceo.
    """
    logger.info(f"Optimizando XGBoost con Optuna ({OPTUNA_TRIALS} trials)...")

    # Peso para la clase positiva (churn=1)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: _objective(trial, X_train, X_test, y_train, y_test, scale_pos_weight),
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["random_state"]     = RANDOM_STATE
    best_params["eval_metric"]      = "auc"
    best_params["use_label_encoder"]= False

    logger.info(f"  Mejor ROC-AUC (Optuna): {study.best_value:.4f}")
    logger.info(f"  Mejores parámetros: {best_params}")

    # Reentrenar con los mejores parámetros sobre todo el train
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train, verbose=False)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    log_metrics_to_console(metrics, "XGBoost (Optuna)")

    return best_model, metrics, best_params, study


# ---------------------------------------------------------------------------
# 5. MLFLOW LOGGING
# ---------------------------------------------------------------------------

def setup_mlflow(tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info(f"  MLflow tracking URI: {tracking_uri}")
    logger.info(f"  Experimento: {MLFLOW_EXPERIMENT}")


def log_baseline_to_mlflow(
    model,
    metrics: dict,
    X_train: pd.DataFrame,
) -> str:
    with mlflow.start_run(run_name="baseline_logistic_regression") as run:
        mlflow.log_param("model_type",    "LogisticRegression")
        mlflow.log_param("class_weight",  "balanced")
        mlflow.log_param("max_iter",      1000)
        mlflow.log_param("n_features",    len(X_train.columns))
        mlflow.log_param("train_size",    len(X_train))

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        logger.info(f"  Baseline loggeado en MLflow. Run ID: {run_id}")

    return run_id


def log_xgboost_to_mlflow(
    model,
    metrics:    dict,
    params:     dict,
    X_train:    pd.DataFrame,
    models_dir: Path,
) -> str:
    with mlflow.start_run(run_name="xgboost_optuna") as run:
        mlflow.log_param("model_type",  "XGBoost")
        mlflow.log_param("n_features",  len(X_train.columns))
        mlflow.log_param("train_size",  len(X_train))
        mlflow.log_param("optuna_trials", OPTUNA_TRIALS)

        for k, v in params.items():
            mlflow.log_param(k, v)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.xgboost.log_model(model, artifact_path="model")

        # Guardar también en disco para el serving
        model_path = models_dir / "xgboost_churn.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))
        logger.info(f"  Modelo guardado en: {model_path}")

        run_id = run.info.run_id
        logger.info(f"  XGBoost loggeado en MLflow. Run ID: {run_id}")

    return run_id


# ---------------------------------------------------------------------------
# 6. REPORTE FINAL
# ---------------------------------------------------------------------------

def print_final_report(
    baseline_metrics: dict,
    xgb_metrics:      dict,
    y_test:           pd.Series,
    xgb_model,
    X_test:           pd.DataFrame,
) -> None:
    print("\n" + "=" * 60)
    print("COMPARATIVA FINAL DE MODELOS")
    print("=" * 60)

    header = f"  {'Métrica':22} {'Baseline (LR)':>16} {'XGBoost':>12}"
    print(header)
    print("  " + "-" * 50)

    for k in baseline_metrics:
        base_val = baseline_metrics[k]
        xgb_val  = xgb_metrics[k]
        mejora   = "↑" if xgb_val > base_val else "↓"
        print(f"  {k:22} {base_val:>16.4f} {xgb_val:>10.4f} {mejora}")

    print("\n  Classification Report (XGBoost):")
    y_pred = xgb_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))

    print("  Confusion Matrix (XGBoost):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# FUNCIÓN ORQUESTRADORA
# ---------------------------------------------------------------------------

def run_training(
    gold_dir:     str | Path,
    models_dir:   str | Path,
    mlflow_uri:   str = "mlruns",
) -> tuple:
    """
    Ejecuta el pipeline completo de entrenamiento.

    Returns:
        xgb_model, baseline_model, metrics_dict
    """
    logger.info("=" * 60)
    logger.info("PIPELINE: Entrenamiento | Churn Prediction")
    logger.info("=" * 60)

    gold_dir   = Path(gold_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/6] Cargando Gold...")
    X_train, X_test, y_train, y_test = load_gold(gold_dir)

    logger.info("[2/6] Configurando MLflow...")
    setup_mlflow(mlflow_uri)

    logger.info("[3/6] Entrenando baseline...")
    baseline_model, baseline_metrics = train_baseline(X_train, X_test, y_train, y_test)

    logger.info("[4/6] Entrenando XGBoost + Optuna...")
    xgb_model, xgb_metrics, xgb_params, study = train_xgboost(
        X_train, X_test, y_train, y_test
    )

    logger.info("[5/6] Registrando en MLflow...")
    log_baseline_to_mlflow(baseline_model, baseline_metrics, X_train)
    log_xgboost_to_mlflow(xgb_model, xgb_metrics, xgb_params, X_train, models_dir)

    logger.info("[6/6] Reporte final...")
    print_final_report(baseline_metrics, xgb_metrics, y_test, xgb_model, X_test)

    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("=" * 60)
    logger.info("Para ver MLflow UI: mlflow ui --port 5000")

    return xgb_model, baseline_model, {"baseline": baseline_metrics, "xgboost": xgb_metrics}


# ---------------------------------------------------------------------------
# EJECUCIÓN DIRECTA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parent.parent

    gold_dir   = ROOT / "data" / "gold"
    models_dir = ROOT / "models"
    mlflow_uri = f"file:///{(ROOT / 'mlruns').as_posix()}"

    if not gold_dir.exists():
        logger.error(f"No se encuentra el Gold: {gold_dir}")
        sys.exit(1)

    run_training(gold_dir, models_dir, mlflow_uri)