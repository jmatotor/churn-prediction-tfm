"""
pipelines/05_evaluate.py
=========================
Ejecuta la evaluación completa del mejor modelo.
Uso: python pipelines/05_evaluate.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import run_evaluation

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

MODELS_DIR  = ROOT / "models"
GOLD_DIR    = ROOT / "data" / "gold"
REPORTS_DIR = ROOT / "reports" / "evaluation"

# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if not (MODELS_DIR / "best_model.pkl").exists():
        print("ERROR: No se encuentra best_model.pkl")
        print("Ejecuta primero: python pipelines/04_train.py")
        sys.exit(1)

    metrics = run_evaluation(
        models_dir=MODELS_DIR,
        gold_dir=GOLD_DIR,
        reports_dir=REPORTS_DIR,
    )

    print(f"\nResumen:")
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision  : {metrics['avg_precision']:.4f}")
    print(f"  F1             : {metrics['f1']:.4f}")
    print(f"  Gráficos en    : {REPORTS_DIR}")