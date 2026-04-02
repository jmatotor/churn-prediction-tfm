"""
src/eda_silver.py
=================
Análisis exploratorio de datos (EDA) sobre la capa Silver.
Genera gráficos en reports/eda/ y un resumen de insights por consola.

Estructura del análisis:
  1. Overview general del dataset
  2. Variable target: Churn
  3. Variables numéricas vs Churn
  4. Variables categóricas vs Churn
  5. Correlaciones
  6. Resumen de insights para feature engineering
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN VISUAL
# ---------------------------------------------------------------------------

CHURN_PALETTE  = {"Yes": "#E24B4A", "No": "#378ADD"}
CHURN_COLORS   = [CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]]
PLOT_STYLE     = "whitegrid"
FIGURE_DPI     = 120

def _setup_style():
    sns.set_theme(style=PLOT_STYLE, palette="muted", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })

def _save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    path = output_dir / filename
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info(f"  Gráfico guardado: {path.name}")


# ---------------------------------------------------------------------------
# 1. OVERVIEW
# ---------------------------------------------------------------------------

def overview(df: pd.DataFrame) -> dict:
    """
    Imprime un resumen general del dataset Silver:
    dimensiones, tipos, nulls y estadísticas básicas.
    """
    logger.info("--- [1] OVERVIEW ---")

    biz_cols = [c for c in df.columns if not c.startswith("_")]
    df_biz   = df[biz_cols]

    num_cols = df_biz.select_dtypes(include="number").columns.tolist()
    cat_cols = df_biz.select_dtypes(include="object").columns.tolist()

    print("\n=== OVERVIEW DEL DATASET SILVER ===")
    print(f"  Filas:               {len(df):,}")
    print(f"  Columnas negocio:    {len(biz_cols)}")
    print(f"  Numéricas:           {len(num_cols)} → {num_cols}")
    print(f"  Categóricas:         {len(cat_cols)} → {cat_cols}")
    print(f"  Nulos totales:       {df_biz.isnull().sum().sum()}")

    print("\n--- Estadísticas numéricas ---")
    print(df_biz[num_cols].describe().round(2).to_string())

    print("\n--- Valores únicos por columna categórica ---")
    for col in cat_cols:
        print(f"  {col:25}: {df_biz[col].nunique()} valores → {df_biz[col].unique().tolist()}")

    stats = {
        "n_rows": len(df),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }
    return stats


# ---------------------------------------------------------------------------
# 2. VARIABLE TARGET: CHURN
# ---------------------------------------------------------------------------

def analyze_target(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Distribución de la variable target Churn.
    Señala el desbalanceo de clases.
    """
    logger.info("--- [2] VARIABLE TARGET ---")

    counts = df["churn"].value_counts()
    pcts   = df["churn"].value_counts(normalize=True)

    print("\n=== VARIABLE TARGET: CHURN ===")
    for val in counts.index:
        print(f"  {val}: {counts[val]:,} ({pcts[val]:.1%})")
    print(f"\n  → Dataset desbalanceado: ratio No/Yes = {counts['No']/counts['Yes']:.1f}x")
    print("  → Estrategia recomendada: class_weight='balanced' o SMOTE en modelado")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Barras con conteos
    bars = axes[0].bar(counts.index, counts.values,
                       color=[CHURN_PALETTE[k] for k in counts.index],
                       edgecolor="white", linewidth=1.5)
    axes[0].set_title("Distribución de Churn (conteos)", fontweight="bold")
    axes[0].set_ylabel("Nº clientes")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 40, f"{val:,}",
                     ha="center", fontsize=11)

    # Tarta con porcentajes
    axes[1].pie(
        pcts.values,
        labels=[f"{k}\n{v:.1%}" for k, v in pcts.items()],
        colors=[CHURN_PALETTE[k] for k in pcts.index],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Proporción de Churn", fontweight="bold")

    plt.suptitle("Variable Target — Churn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "01_target_churn.png")

    return {"churn_rate": pcts.get("Yes", 0), "class_ratio": counts["No"] / counts["Yes"]}


# ---------------------------------------------------------------------------
# 3. VARIABLES NUMÉRICAS
# ---------------------------------------------------------------------------

def analyze_numerical(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Analiza tenure_months, monthly_charges y total_charges contra Churn.
    Genera histogramas, boxplots y tabla de medias por grupo.
    """
    logger.info("--- [3] VARIABLES NUMÉRICAS ---")

    num_cols = ["tenure_months", "monthly_charges", "total_charges"]

    # Tabla de medias por grupo churn
    print("\n=== VARIABLES NUMÉRICAS POR GRUPO CHURN ===")
    stats_by_churn = df.groupby("churn")[num_cols].mean().round(2)
    print(stats_by_churn.to_string())

    # Diferencias relativas (útil para detectar features discriminativas)
    print("\n  Diferencia relativa (Churn Yes vs No):")
    for col in num_cols:
        mean_yes = df[df["churn"] == "Yes"][col].mean()
        mean_no  = df[df["churn"] == "No"][col].mean()
        diff_pct = (mean_yes - mean_no) / mean_no * 100
        direction = "↑" if diff_pct > 0 else "↓"
        print(f"  {col:25}: {direction} {abs(diff_pct):.1f}% en churners")

    # --- Histogramas por churn ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, num_cols):
        for churn_val, color in CHURN_PALETTE.items():
            subset = df[df["churn"] == churn_val][col]
            ax.hist(subset, bins=35, alpha=0.65,
                    label=f"Churn={churn_val}", color=color, edgecolor="none")
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=9)
    plt.suptitle("Distribución de variables numéricas por Churn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "02_numerical_histograms.png")

    # --- Boxplots por churn ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col in zip(axes, num_cols):
        groups = [df[df["churn"] == v][col].values for v in ["No", "Yes"]]
        bp = ax.boxplot(groups, labels=["No", "Yes"], patch_artist=True,
                        medianprops={"color": "white", "linewidth": 2})
        for patch, color in zip(bp["boxes"], CHURN_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("Churn")
    plt.suptitle("Boxplots de variables numéricas por Churn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "03_numerical_boxplots.png")

    insights = {col: {
        "mean_churn_yes": df[df["churn"] == "Yes"][col].mean(),
        "mean_churn_no":  df[df["churn"] == "No"][col].mean(),
    } for col in num_cols}

    return insights


# ---------------------------------------------------------------------------
# 4. VARIABLES CATEGÓRICAS
# ---------------------------------------------------------------------------

def analyze_categorical(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Para cada variable categórica calcula la tasa de churn por categoría.
    Genera gráficos de barras apiladas con porcentajes.
    """
    logger.info("--- [4] VARIABLES CATEGÓRICAS ---")

    cat_cols = [
        "gender", "senior_citizen", "partner", "dependents",
        "phone_service", "multiple_lines", "internet_service",
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
        "contract", "paperless_billing", "payment_method",
    ]

    # Tasa de churn por categoría
    print("\n=== TASA DE CHURN POR VARIABLE CATEGÓRICA ===")
    churn_rates = {}
    for col in cat_cols:
        col_data = df[col].astype(str)
        rate = df.groupby(col_data)["churn"].apply(
            lambda x: (x == "Yes").mean()
        ).sort_values(ascending=False)
        churn_rates[col] = rate
        print(f"\n  {col}:")
        for cat, val in rate.items():
            print(f"    {str(cat):30}: {val:.1%}")

    # --- Gráfico 1: variables demográficas y de contrato (más relevantes) ---
    top_cols = ["contract", "internet_service", "payment_method",
                "senior_citizen", "partner", "dependents",
                "paperless_billing", "phone_service"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for ax, col in zip(axes, top_cols):
        col_data = df[col].astype(str)
        ct = df.groupby([col_data, "churn"]).size().unstack(fill_value=0)
        # Reordenar para que No sea primero
        ct = ct[["No", "Yes"]] if "No" in ct.columns else ct
        ct_pct = ct.div(ct.sum(axis=1), axis=0)
        ct_pct.plot(kind="bar", ax=ax, stacked=True,
                    color=CHURN_COLORS, legend=False, rot=25)
        ax.set_title(col, fontweight="bold", fontsize=10)
        ax.set_ylabel("Proporción")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_xlabel("")
        # Añadir la tasa de churn encima de cada barra
        for i, (idx, row) in enumerate(ct_pct.iterrows()):
            churn_pct = row.get("Yes", 0)
            ax.text(i, 1.02, f"{churn_pct:.0%}",
                    ha="center", fontsize=8, color=CHURN_PALETTE["Yes"],
                    fontweight="bold")

    # Leyenda en el último eje
    axes[-1].legend(["No churn", "Churn"], loc="center", fontsize=10,
                    frameon=True, title="Churn")

    plt.suptitle("Tasa de Churn por variable categórica", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "04_categorical_churn_rates.png")

    # --- Gráfico 2: columnas de servicios ---
    service_cols = ["online_security", "online_backup", "device_protection",
                    "tech_support", "streaming_tv", "streaming_movies", "multiple_lines"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, service_cols):
        col_data = df[col].astype(str)
        rate = df.groupby(col_data)["churn"].apply(lambda x: (x == "Yes").mean())
        colors = [CHURN_PALETTE["Yes"] if v > 0.3 else CHURN_PALETTE["No"]
                  for v in rate.values]
        bars = ax.bar(rate.index, rate.values, color=colors, alpha=0.8)
        ax.set_title(col, fontweight="bold", fontsize=10)
        ax.set_ylabel("Churn rate")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.axhline(y=df["churn"].eq("Yes").mean(), color="gray",
                   linestyle="--", linewidth=1, alpha=0.7, label="Media global")
        ax.set_xlabel("")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.0%}", ha="center", fontsize=9)

    # Ocultar eje vacío
    axes[-1].set_visible(False)

    plt.suptitle("Churn rate por columnas de servicios", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "05_services_churn_rates.png")

    return churn_rates


# ---------------------------------------------------------------------------
# 5. CORRELACIONES
# ---------------------------------------------------------------------------

def analyze_correlations(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Calcula correlaciones entre variables numéricas y el target codificado.
    Incluye senior_citizen (ya es int) y churn codificado como 0/1.
    """
    logger.info("--- [5] CORRELACIONES ---")

    df_corr = df.copy()
    df_corr["churn_num"] = (df_corr["churn"] == "Yes").astype(int)

    num_cols_corr = ["tenure_months", "monthly_charges", "total_charges",
                     "senior_citizen", "churn_num"]

    corr = df_corr[num_cols_corr].corr()

    print("\n=== CORRELACIÓN CON CHURN (variables numéricas) ===")
    churn_corr = corr["churn_num"].drop("churn_num").sort_values(key=abs, ascending=False)
    for col, val in churn_corr.items():
        direction = "positiva" if val > 0 else "negativa"
        print(f"  {col:25}: {val:+.3f} ({direction})")

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, ax=ax,
        linewidths=0.5, square=True, mask=mask,
        annot_kws={"size": 10},
    )
    ax.set_title("Matriz de correlación (variables numéricas + churn)", fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "06_correlation_matrix.png")


# ---------------------------------------------------------------------------
# 6. RESUMEN DE INSIGHTS
# ---------------------------------------------------------------------------

def print_insights_summary(churn_rates: dict) -> None:
    """
    Imprime los insights más relevantes para guiar el feature engineering en Gold.
    """
    print("\n" + "=" * 60)
    print("INSIGHTS CLAVE PARA FEATURE ENGINEERING (→ Gold)")
    print("=" * 60)

    insights = [
        ("Contract",        "contract",         "Month-to-month"),
        ("Internet",        "internet_service",  "Fiber optic"),
        ("Online security", "online_security",   "No"),
        ("Tech support",    "tech_support",      "No"),
        ("Payment",         "payment_method",    "Electronic check"),
    ]

    for label, col, high_risk_cat in insights:
        if col in churn_rates:
            rate = churn_rates[col].get(high_risk_cat, None)
            if rate is not None:
                print(f"  {label:20}: '{high_risk_cat}' → {rate:.1%} churn rate")

    print("\n  Recomendaciones para Gold:")
    print("  · Crear feature 'is_month_to_month' (contract)")
    print("  · Crear feature 'n_services_contratados' (suma de servicios activos)")
    print("  · Crear feature 'charges_per_month_ratio' (total_charges / tenure_months)")
    print("  · Encodear contract y internet_service con target encoding o ordinal")
    print("  · Tratar desbalanceo: SMOTE o class_weight en modelado")
    print("=" * 60)


# ---------------------------------------------------------------------------
# FUNCIÓN ORQUESTRADORA
# ---------------------------------------------------------------------------

def run_eda(silver_path: str | Path, reports_dir: str | Path) -> None:
    """
    Ejecuta el EDA completo sobre el Silver y guarda los resultados en reports_dir.
    """
    silver_path  = Path(silver_path)
    reports_dir  = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()

    logger.info("=" * 60)
    logger.info("EDA: Silver — Telco Customer Churn")
    logger.info("=" * 60)

    # Carga
    logger.info("Cargando Silver...")
    df = pd.read_parquet(silver_path)
    logger.info(f"  {len(df):,} registros cargados.")

    # Análisis
    overview(df)
    target_stats   = analyze_target(df, reports_dir)
    num_insights   = analyze_numerical(df, reports_dir)
    churn_rates    = analyze_categorical(df, reports_dir)
    analyze_correlations(df, reports_dir)
    print_insights_summary(churn_rates)

    logger.info("=" * 60)
    logger.info(f"EDA completado. Gráficos en: {reports_dir}")
    logger.info("=" * 60)