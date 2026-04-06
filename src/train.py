"""
src/train.py
============
Entrenamiento con selección automática de modelo.

Modelos comparados:
  - Logistic Regression  (baseline interpretable)
  - Random Forest        (ensemble sin boosting)
  - XGBoost              (gradient boosting)
  - LightGBM             (gradient boosting rápido)

Flujo:
  1. Carga Gold (X_train, X_test, y_train, y_test)
  2. Cada modelo pasa por Optuna (búsqueda de hiperparámetros)
  3. Comparativa de métricas en test
  4. El mejor modelo (ROC-AUC) se registra en MLflow y se guarda en models/
  5. Los demás se registran también para trazabilidad

Diseñado para funcionar con cualquier dataset binario de churn,
no solo Telco. El único requisito es que Gold tenga el target en 'churn'.
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
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
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

RANDOM_STATE       = 42
OPTUNA_TRIALS      = 30
MLFLOW_EXPERIMENT  = "churn-prediction-tfm"
METRIC_TO_OPTIMIZE = "roc_auc"


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
    return {
        "roc_auc":       round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
        "f1":            round(f1_score(y_true, y_pred), 4),
        "precision":     round(precision_score(y_true, y_pred), 4),
        "recall":        round(recall_score(y_true, y_pred), 4),
    }


# ---------------------------------------------------------------------------
# 3. OBJETIVOS OPTUNA
# ---------------------------------------------------------------------------

def _get_scale_pos_weight(y_train: pd.Series) -> float:
    return (y_train == 0).sum() / (y_train == 1).sum()


def _objective_lr(trial, X_train, X_test, y_train, y_test):
    params = {
        "C":            trial.suggest_float("C", 1e-3, 10.0, log=True),
        "solver":       trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        "max_iter":     1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def _objective_rf(trial, X_train, X_test, y_train, y_test):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 400),
        "max_depth":         trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight":      "balanced",
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def _objective_xgb(trial, X_train, X_test, y_train, y_test, scale_pos_weight):
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
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def _objective_lgbm(trial, X_train, X_test, y_train, y_test, scale_pos_weight):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 80),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        "scale_pos_weight":  scale_pos_weight,
        "random_state":      RANDOM_STATE,
        "verbose":           -1,
        "n_jobs":            -1,
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


# ---------------------------------------------------------------------------
# 4. ENTRENAMIENTO DE TODOS LOS MODELOS
# ---------------------------------------------------------------------------

def _run_study(objective_fn, n_trials: int) -> optuna.Study:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study


def train_all_models(X_train, X_test, y_train, y_test) -> dict:
    """
    Entrena los 4 modelos con Optuna y devuelve resultados comparables.
    Estructura por modelo: {model, params, metrics, study}
    """
    scale_pos_weight = _get_scale_pos_weight(y_train)
    results = {}

    # Logistic Regression
    logger.info("  Optimizando Logistic Regression...")
    study = _run_study(
        lambda t: _objective_lr(t, X_train, X_test, y_train, y_test),
        OPTUNA_TRIALS,
    )
    model = LogisticRegression(
        **{k: v for k, v in study.best_params.items() if k in ("C", "solver")},
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results["logistic_regression"] = {
        "model": model, "params": study.best_params,
        "metrics": compute_metrics(y_test, y_pred, y_prob), "study": study,
    }
    logger.info(f"    ROC-AUC: {results['logistic_regression']['metrics']['roc_auc']:.4f}")

    # Random Forest
    logger.info("  Optimizando Random Forest...")
    study = _run_study(
        lambda t: _objective_rf(t, X_train, X_test, y_train, y_test),
        OPTUNA_TRIALS,
    )
    model = RandomForestClassifier(
        **study.best_params, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results["random_forest"] = {
        "model": model, "params": study.best_params,
        "metrics": compute_metrics(y_test, y_pred, y_prob), "study": study,
    }
    logger.info(f"    ROC-AUC: {results['random_forest']['metrics']['roc_auc']:.4f}")

    # XGBoost
    logger.info("  Optimizando XGBoost...")
    study = _run_study(
        lambda t: _objective_xgb(t, X_train, X_test, y_train, y_test, scale_pos_weight),
        OPTUNA_TRIALS,
    )
    best_params_xgb = {
        **study.best_params,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "eval_metric": "auc",
        "use_label_encoder": False,
    }
    model = XGBClassifier(**best_params_xgb)
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results["xgboost"] = {
        "model": model, "params": best_params_xgb,
        "metrics": compute_metrics(y_test, y_pred, y_prob), "study": study,
    }
    logger.info(f"    ROC-AUC: {results['xgboost']['metrics']['roc_auc']:.4f}")

    # LightGBM
    logger.info("  Optimizando LightGBM...")
    study = _run_study(
        lambda t: _objective_lgbm(t, X_train, X_test, y_train, y_test, scale_pos_weight),
        OPTUNA_TRIALS,
    )
    best_params_lgbm = {
        **study.best_params,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "verbose": -1, "n_jobs": -1,
    }
    model = LGBMClassifier(**best_params_lgbm)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results["lightgbm"] = {
        "model": model, "params": best_params_lgbm,
        "metrics": compute_metrics(y_test, y_pred, y_prob), "study": study,
    }
    logger.info(f"    ROC-AUC: {results['lightgbm']['metrics']['roc_auc']:.4f}")

    return results


# ---------------------------------------------------------------------------
# 5. SELECCIÓN DEL MEJOR MODELO
# ---------------------------------------------------------------------------

def select_best_model(results: dict) -> tuple:
    best_name = max(results, key=lambda k: results[k]["metrics"][METRIC_TO_OPTIMIZE])
    logger.info(
        f"  Mejor modelo: {best_name} "
        f"(ROC-AUC: {results[best_name]['metrics'][METRIC_TO_OPTIMIZE]:.4f})"
    )
    return best_name, results[best_name]


# ---------------------------------------------------------------------------
# 6. MLFLOW LOGGING
# ---------------------------------------------------------------------------

def log_all_to_mlflow(results: dict, best_name: str, X_train: pd.DataFrame, models_dir: Path) -> None:
    mlflow.set_tracking_uri(f"file:///{(models_dir.parent / 'mlruns').as_posix()}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for model_name, result in results.items():
        is_best  = model_name == best_name
        run_name = f"{model_name}{'__BEST' if is_best else ''}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("best_model", str(is_best).lower())
            mlflow.set_tag("model_type", model_name)
            mlflow.log_param("n_features",    len(X_train.columns))
            mlflow.log_param("train_size",    len(X_train))
            mlflow.log_param("optuna_trials", OPTUNA_TRIALS)

            for k, v in result["params"].items():
                mlflow.log_param(k, v)
            for k, v in result["metrics"].items():
                mlflow.log_metric(k, v)

            model = result["model"]
            if model_name == "xgboost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            if is_best:
                model_path = models_dir / "best_model.pkl"
                joblib.dump({"model": model, "model_name": model_name}, model_path)
                mlflow.log_artifact(str(model_path))
                logger.info(f"  Mejor modelo guardado en: {model_path}")


# ---------------------------------------------------------------------------
# 7. REPORTE FINAL
# ---------------------------------------------------------------------------

def print_final_report(results: dict, best_name: str, y_test, X_test) -> None:
    print("\n" + "=" * 70)
    print("COMPARATIVA DE MODELOS")
    print("=" * 70)

    metric_keys   = ["roc_auc", "avg_precision", "f1", "precision", "recall"]
    print(f"  {'Modelo':25}" + "".join(f"{m:>14}" for m in metric_keys))
    print("  " + "-" * 65)

    for name, result in sorted(results.items(),
                                key=lambda x: x[1]["metrics"]["roc_auc"],
                                reverse=True):
        tag = " <- MEJOR" if name == best_name else ""
        row = f"  {name:25}" + "".join(f"{result['metrics'][m]:>14.4f}" for m in metric_keys) + tag
        print(row)

    print("\n  Classification Report (mejor modelo):")
    best_model = results[best_name]["model"]
    y_pred     = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# FUNCIÓN ORQUESTRADORA
# ---------------------------------------------------------------------------

def run_training(gold_dir: str | Path, models_dir: str | Path, mlflow_uri: str = "mlruns") -> tuple:
    logger.info("=" * 60)
    logger.info("PIPELINE: Entrenamiento y selección de modelo")
    logger.info("=" * 60)

    gold_dir   = Path(gold_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/5] Cargando Gold...")
    X_train, X_test, y_train, y_test = load_gold(gold_dir)

    logger.info("[2/5] Entrenando y optimizando 4 modelos con Optuna...")
    results = train_all_models(X_train, X_test, y_train, y_test)

    logger.info("[3/5] Seleccionando mejor modelo...")
    best_name, best_result = select_best_model(results)

    logger.info("[4/5] Registrando todos los modelos en MLflow...")
    log_all_to_mlflow(results, best_name, X_train, models_dir)

    logger.info("[5/5] Reporte final...")
    print_final_report(results, best_name, y_test, X_test)

    logger.info("=" * 60)
    logger.info(f"COMPLETADO — Mejor modelo: {best_name}")
    logger.info("Para ver MLflow UI: mlflow ui --port 5000")
    logger.info("=" * 60)

    return best_result["model"], best_name, results


# ---------------------------------------------------------------------------
# EJECUCIÓN DIRECTA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parent.parent

    gold_dir   = ROOT / "data" / "gold"
    models_dir = ROOT / "models"

    if not gold_dir.exists():
        logger.error(f"No se encuentra el Gold: {gold_dir}")
        sys.exit(1)

    best_model, best_name, results = run_training(gold_dir, models_dir)

    print(f"\nModelo ganador : {best_name}")
    print(f"Guardado en    : {models_dir / 'best_model.pkl'}")
    print(f"ROC-AUC        : {results[best_name]['metrics']['roc_auc']:.4f}")