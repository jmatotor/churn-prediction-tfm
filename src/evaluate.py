"""
src/evaluate.py
===============
Evaluación completa del mejor modelo entrenado.

Genera en reports/evaluation/:
  01_roc_curve.png           - Curva ROC con AUC
  02_precision_recall.png    - Curva Precision-Recall con AP
  03_confusion_matrix.png    - Matriz de confusión normalizada y absoluta
  04_shap_beeswarm.png       - Importancia global de features (SHAP)
  05_shap_bar.png            - Top features por valor SHAP medio absoluto
  06_shap_waterfall.png      - Explicación local de un cliente concreto
  07_threshold_analysis.png  - Precision/Recall/F1 vs umbral de decisión
"""

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

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

CHURN_PALETTE = {"Yes": "#E24B4A", "No": "#378ADD"}
FIGURE_DPI    = 120
RANDOM_STATE  = 42


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _setup_style():
    plt.rcParams.update({
        "figure.dpi":          FIGURE_DPI,
        "figure.facecolor":    "white",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "savefig.bbox":        "tight",
        "savefig.facecolor":   "white",
        "font.size":           11,
    })


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    path = output_dir / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info(f"  Guardado: {filename}")


# ---------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------

def load_model_and_data(models_dir: Path, gold_dir: Path) -> tuple:
    """
    Carga el mejor modelo guardado por train.py y los datos de test.
    """
    model_path = models_dir / "best_model.pkl"
    artifact   = joblib.load(model_path)
    model      = artifact["model"]
    model_name = artifact["model_name"]

    X_test = pd.read_parquet(gold_dir / "X_test.parquet")
    y_test = pd.read_parquet(gold_dir / "y_test.parquet").squeeze()

    logger.info(f"  Modelo cargado: {model_name}")
    logger.info(f"  Test set: {len(X_test):,} filas | Churn rate: {y_test.mean():.1%}")

    return model, model_name, X_test, y_test


# ---------------------------------------------------------------------------
# 2. PREDICCIONES
# ---------------------------------------------------------------------------

def get_predictions(model, X_test: pd.DataFrame) -> tuple:
    """
    Devuelve predicciones binarias y probabilidades.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


# ---------------------------------------------------------------------------
# 3. MÉTRICAS POR CONSOLA
# ---------------------------------------------------------------------------

def print_metrics(y_test, y_pred, y_prob, model_name: str) -> dict:
    metrics = {
        "roc_auc":       round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "f1":            round(f1_score(y_test, y_pred), 4),
        "precision":     round(precision_score(y_test, y_pred), 4),
        "recall":        round(recall_score(y_test, y_pred), 4),
    }

    print(f"\n=== MÉTRICAS DE EVALUACIÓN — {model_name.upper()} ===")
    for k, v in metrics.items():
        print(f"  {k:20}: {v:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix:")
    print(f"    TN={tn:4d}  FP={fp:4d}  (de {tn+fp} negativos reales)")
    print(f"    FN={fn:4d}  TP={tp:4d}  (de {fn+tp} positivos reales)")
    print(f"\n  Interpretación de negocio:")
    print(f"    · Clientes en riesgo detectados correctamente (TP): {tp}")
    print(f"    · Clientes en riesgo NO detectados (FN — coste alto): {fn}")
    print(f"    · Alertas falsas generadas (FP — coste bajo): {fp}")

    return metrics


# ---------------------------------------------------------------------------
# 4. CURVA ROC
# ---------------------------------------------------------------------------

def plot_roc_curve(y_test, y_prob, model_name: str, output_dir: Path) -> None:
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    # Punto de máximo Youden Index (mejor umbral en ROC)
    youden_idx     = np.argmax(tpr - fpr)
    best_threshold = thresholds[youden_idx]
    best_fpr       = fpr[youden_idx]
    best_tpr       = tpr[youden_idx]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color=CHURN_PALETTE["Yes"], linewidth=2.5,
            label=f"{model_name} (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle="--",
            label="Random classifier (AUC = 0.500)")

    # Marcar el mejor umbral
    ax.scatter(best_fpr, best_tpr, color=CHURN_PALETTE["Yes"],
               s=100, zorder=5, label=f"Mejor umbral = {best_threshold:.2f}")
    ax.annotate(
        f"  umbral={best_threshold:.2f}\n  TPR={best_tpr:.2f}, FPR={best_fpr:.2f}",
        xy=(best_fpr, best_tpr), fontsize=9, color="gray",
    )

    ax.fill_between(fpr, tpr, alpha=0.08, color=CHURN_PALETTE["Yes"])
    ax.set_xlabel("False Positive Rate (1 - Especificidad)")
    ax.set_ylabel("True Positive Rate (Sensibilidad)")
    ax.set_title("Curva ROC", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    _save(fig, output_dir, "01_roc_curve.png")

    print(f"\n  ROC-AUC: {auc_score:.4f}")
    print(f"  Mejor umbral (Youden): {best_threshold:.3f} → TPR={best_tpr:.3f}, FPR={best_fpr:.3f}")


# ---------------------------------------------------------------------------
# 5. CURVA PRECISION-RECALL
# ---------------------------------------------------------------------------

def plot_precision_recall(y_test, y_prob, model_name: str, output_dir: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    ap_score   = average_precision_score(y_test, y_prob)
    baseline   = y_test.mean()

    # F1 máximo
    f1_scores  = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx   = np.argmax(f1_scores)
    best_thr   = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(recall, precision, color=CHURN_PALETTE["Yes"], linewidth=2.5,
            label=f"{model_name} (AP = {ap_score:.3f})")
    ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline (churn rate = {baseline:.2%})")
    ax.scatter(recall[best_idx], precision[best_idx],
               color=CHURN_PALETTE["Yes"], s=100, zorder=5,
               label=f"Mejor F1 (umbral={best_thr:.2f})")

    ax.fill_between(recall, precision, alpha=0.08, color=CHURN_PALETTE["Yes"])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    _save(fig, output_dir, "02_precision_recall.png")

    print(f"\n  Average Precision: {ap_score:.4f}")
    print(f"  Umbral óptimo F1:  {best_thr:.3f} → Precision={precision[best_idx]:.3f}, Recall={recall[best_idx]:.3f}")


# ---------------------------------------------------------------------------
# 6. MATRIZ DE CONFUSIÓN
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_test, y_pred, output_dir: Path) -> None:
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels    = ["No churn", "Churn"]

    for ax, matrix, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Matriz de confusión (conteos)", "Matriz de confusión (normalizada)"],
        ["d", ".2%"],
    ):
        im = ax.imshow(matrix, interpolation="nearest",
                       cmap=plt.cm.Blues, vmin=0)
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")

        thresh = matrix.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                text = f"{val:{fmt}}" if fmt == "d" else f"{val:.1%}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=13, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    _save(fig, axes[0].figure, "03_confusion_matrix.png")


def _save(fig_or_ax, output_dir_or_fig, filename_or_str):
    """Overload para manejar la llamada desde plot_confusion_matrix."""
    if isinstance(fig_or_ax, plt.Figure):
        fig  = fig_or_ax
        path = output_dir_or_fig / filename_or_str
    else:
        fig  = output_dir_or_fig
        path = Path(filename_or_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info(f"  Guardado: {path.name}")


# Redefinir _save limpio después del overload
def _save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    path = output_dir / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info(f"  Guardado: {filename}")


def plot_confusion_matrix(y_test, y_pred, output_dir: Path) -> None:
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
    labels  = ["No churn", "Churn"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, matrix, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Conteos", "Normalizada (por fila)"],
        ["d", ".1%"],
    ):
        im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        thresh = matrix.max() / 2.0
        for i in range(2):
            for j in range(2):
                val  = matrix[i, j]
                text = str(val) if fmt == "d" else f"{val:.1%}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Matriz de Confusión", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "03_confusion_matrix.png")


# ---------------------------------------------------------------------------
# 7. ANÁLISIS DE UMBRAL
# ---------------------------------------------------------------------------

def plot_threshold_analysis(y_test, y_prob, output_dir: Path) -> None:
    """
    Muestra cómo varían Precision, Recall y F1 al cambiar el umbral de decisión.
    Útil para justificar si se optimiza para no perder churners (recall)
    o para no saturar de alertas (precision).
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        if y_pred_t.sum() == 0:
            precisions.append(1.0)
            recalls.append(0.0)
            f1s.append(0.0)
        else:
            precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
            recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
            f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

    best_f1_idx = np.argmax(f1s)
    best_thr    = thresholds[best_f1_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#378ADD", linewidth=2)
    ax.plot(thresholds, recalls,    label="Recall",    color="#E24B4A", linewidth=2)
    ax.plot(thresholds, f1s,        label="F1",        color="#1D9E75", linewidth=2)
    ax.axvline(x=best_thr, color="gray", linestyle="--", linewidth=1.2,
               label=f"Mejor F1 (umbral={best_thr:.2f})")
    ax.axvline(x=0.5, color="lightgray", linestyle=":", linewidth=1,
               label="Umbral por defecto (0.5)")

    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Umbral de decisión", fontweight="bold")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    _save(fig, output_dir, "07_threshold_analysis.png")

    print(f"\n  Umbral óptimo para F1: {best_thr:.3f}")
    print(f"    Precision: {precisions[best_f1_idx]:.3f}")
    print(f"    Recall:    {recalls[best_f1_idx]:.3f}")
    print(f"    F1:        {f1s[best_f1_idx]:.3f}")
    print(f"  Nota: umbral por defecto=0.5 puede no ser óptimo para churn.")
    print(f"  Si el negocio prioriza no perder churners → bajar umbral (más recall).")


# ---------------------------------------------------------------------------
# 8. SHAP — GLOBAL
# ---------------------------------------------------------------------------

def compute_shap_values(model, X_test: pd.DataFrame, model_name: str):
    """
    Calcula SHAP values adaptándose al tipo de modelo.
    TreeExplainer para tree-based, LinearExplainer para LR.
    """
    logger.info(f"  Calculando SHAP values para {model_name}...")

    if model_name in ("xgboost", "lightgbm", "random_forest"):
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Random Forest devuelve lista [clase0, clase1] — quedarnos con clase positiva
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]

    else:
        # Logistic Regression
        explainer      = shap.LinearExplainer(model, X_test)
        shap_values    = explainer.shap_values(X_test)
        expected_value = explainer.expected_value

    logger.info(f"  SHAP values calculados: {shap_values.shape}")
    return explainer, shap_values, expected_value


def plot_shap_beeswarm(shap_values, X_test: pd.DataFrame, output_dir: Path) -> None:
    """
    Beeswarm plot: muestra la distribución de SHAP values por feature.
    Cada punto es un cliente. Color = valor de la feature.
    Es el gráfico más informativo del SHAP global.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    shap.summary_plot(
        shap_values, X_test,
        plot_type="dot",
        max_display=15,
        show=False,
        plot_size=None,
    )

    plt.title("SHAP Beeswarm — Importancia global de features", fontweight="bold", pad=12)
    plt.tight_layout()
    _save(plt.gcf(), output_dir, "04_shap_beeswarm.png")


def plot_shap_bar(shap_values, X_test: pd.DataFrame, output_dir: Path) -> None:
    """
    Bar plot con el SHAP mean absoluto por feature (importancia media global).
    Más fácil de leer que el beeswarm para comunicar resultados.
    """
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_test.columns,
    ).sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(mean_abs_shap.index, mean_abs_shap.values,
                   color=CHURN_PALETTE["Yes"], alpha=0.8)

    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va="center", fontsize=9)

    ax.set_xlabel("SHAP mean |valor|")
    ax.set_title("Top 15 features por importancia SHAP (global)", fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "05_shap_bar.png")

    print("\n  Top 10 features más importantes (SHAP global):")
    for feat, val in mean_abs_shap.sort_values(ascending=False).head(10).items():
        print(f"    {feat:35}: {val:.4f}")


# ---------------------------------------------------------------------------
# 9. SHAP — LOCAL (cliente individual)
# ---------------------------------------------------------------------------

def plot_shap_waterfall(
    shap_values,
    expected_value,
    X_test:     pd.DataFrame,
    y_test:     pd.Series,
    y_prob:     np.ndarray,
    output_dir: Path,
) -> None:
    """
    Waterfall plot para un cliente concreto: explica por qué el modelo
    le asigna ese score de churn.

    Se elige el cliente con probabilidad de churn más cercana a 0.75
    (cliente de riesgo alto pero no el extremo, más interesante para explicar).
    """
    target_prob = 0.75
    idx = np.argmin(np.abs(y_prob - target_prob))
    prob_real   = y_prob[idx]
    churn_real  = y_test.iloc[idx]

    logger.info(f"  Cliente seleccionado para SHAP local: índice {idx} | prob={prob_real:.3f} | real={'Churn' if churn_real else 'No churn'}")

    shap_explanation = shap.Explanation(
        values        = shap_values[idx],
        base_values   = expected_value,
        data          = X_test.iloc[idx].values,
        feature_names = X_test.columns.tolist(),
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(shap_explanation, max_display=15, show=False)
    plt.title(
        f"SHAP Waterfall — Cliente ejemplo\n"
        f"Score de churn: {prob_real:.1%} | Real: {'Churn' if churn_real else 'No churn'}",
        fontweight="bold", pad=12,
    )
    plt.tight_layout()
    _save(plt.gcf(), output_dir, "06_shap_waterfall.png")


# ---------------------------------------------------------------------------
# FUNCIÓN ORQUESTRADORA
# ---------------------------------------------------------------------------

def run_evaluation(
    models_dir:  str | Path,
    gold_dir:    str | Path,
    reports_dir: str | Path,
) -> dict:
    """
    Ejecuta la evaluación completa del mejor modelo.

    Returns:
        dict con métricas finales
    """
    models_dir  = Path(models_dir)
    gold_dir    = Path(gold_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()

    logger.info("=" * 60)
    logger.info("PIPELINE: Evaluación del modelo")
    logger.info("=" * 60)

    logger.info("[1/5] Cargando modelo y datos...")
    model, model_name, X_test, y_test = load_model_and_data(models_dir, gold_dir)

    logger.info("[2/5] Calculando predicciones y métricas...")
    y_pred, y_prob = get_predictions(model, X_test)
    metrics = print_metrics(y_test, y_pred, y_prob, model_name)

    logger.info("[3/5] Generando gráficos de evaluación...")
    plot_roc_curve(y_test, y_prob, model_name, reports_dir)
    plot_precision_recall(y_test, y_prob, model_name, reports_dir)
    plot_confusion_matrix(y_test, y_pred, reports_dir)
    plot_threshold_analysis(y_test, y_prob, reports_dir)

    logger.info("[4/5] Calculando y graficando SHAP values...")
    explainer, shap_values, expected_value = compute_shap_values(model, X_test, model_name)
    plot_shap_beeswarm(shap_values, X_test, reports_dir)
    plot_shap_bar(shap_values, X_test, reports_dir)
    plot_shap_waterfall(shap_values, expected_value, X_test, y_test, y_prob, reports_dir)

    logger.info("[5/5] Evaluación completada.")
    logger.info("=" * 60)
    logger.info(f"Gráficos en: {reports_dir}")
    logger.info("=" * 60)

    return metrics


# ---------------------------------------------------------------------------
# EJECUCIÓN DIRECTA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parent.parent

    models_dir  = ROOT / "models"
    gold_dir    = ROOT / "data" / "gold"
    reports_dir = ROOT / "reports" / "evaluation"

    if not (models_dir / "best_model.pkl").exists():
        logger.error("No se encuentra best_model.pkl. Ejecuta primero: python pipelines/04_train.py")
        sys.exit(1)

    run_evaluation(models_dir, gold_dir, reports_dir)