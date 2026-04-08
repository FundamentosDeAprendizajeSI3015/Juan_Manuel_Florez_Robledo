"""
Pipeline de Mantenimiento Predictivo — AI4I 2020
Informe 2 | Universidad EAFIT | Aprendizaje Automático

Continuación del Informe 1: clustering → corrección de etiquetas → modelos supervisados.
Split 60/20/20: train / validación / test.

Ejecutar: python pipeline_informe2.py
"""

# ==============================================================
# PASO 0 — Setup
# ==============================================================

import os
import subprocess
import warnings
warnings.filterwarnings("ignore")

DATASET  = "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "ai4i2020.csv")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print("Descargando dataset desde Kaggle...")
    print("  Requiere ~/.kaggle/kaggle.json — ver: https://www.kaggle.com/settings → API")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", DATA_DIR, "--unzip"],
            check=True,
        )
        # Kaggle puede bajar el archivo con otro nombre; lo normalizamos
        for f in os.listdir(DATA_DIR):
            if f.endswith(".csv") and f != "ai4i2020.csv":
                os.rename(os.path.join(DATA_DIR, f), CSV_PATH)
        print(f"Dataset guardado en {CSV_PATH}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR al descargar: {e}")
        raise
else:
    print(f"Dataset ya existe en {CSV_PATH}")

# Detección de GPU; si no hay CUDA el pipeline cae a CPU sin errores
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDispositivo: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("  CUDA no disponible — usando CPU.")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # backend sin ventana para scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, silhouette_score,
    adjusted_rand_score,
)
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import skfuzzy as fuzz

BASE_DIR    = Path(__file__).parent.parent
FIGS_DIR    = BASE_DIR / "outputs" / "figures"
RESULTS_DIR = BASE_DIR / "outputs" / "results"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})
sns.set_theme(style="whitegrid", palette="muted")

RANDOM_STATE = 42
print("\nImports completados.")


# ==============================================================
# PASO 1 — Preprocesamiento (herencia del Informe 1)
# ==============================================================
print("\n" + "=" * 60)
print("PASO 1 — Preprocesamiento")
print("=" * 60)

df_raw = pd.read_csv(CSV_PATH)
print(f"Shape original: {df_raw.shape}")

df = df_raw.copy()

# Eliminamos identificadores que no aportan información al modelo
df.drop(columns=["UDI", "Product ID"], errors="ignore", inplace=True)

# Imputación con mediana (más robusta que la media ante outliers)
numeric_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
for col in numeric_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# Filas duplicadas e inconsistencias físicas (Torque <= 0 = sensor erróneo)
n_antes = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicados eliminados: {n_antes - len(df)}")

n_antes_torque = len(df)
df = df[df["Torque [Nm]"] > 0]
print(f"Filas eliminadas por Torque <= 0: {n_antes_torque - len(df)}")

# One-Hot Encoding de Type (L/M/H) → tres columnas binarias
df = pd.get_dummies(df, columns=["Type"], drop_first=False)

# Los flags de fallo deben ser enteros para scikit-learn
for col in ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]:
    if col in df.columns:
        df[col] = df[col].astype(int)

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_H",
    "Type_L",
    "Type_M",
]
TARGET_COL = "Machine failure"

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

vc = y.value_counts()
pct_positivos = 100 * vc[1] / len(y)
print(f"Shape final: {df.shape}  |  X: {X.shape}")
print(f"Distribución de clases:\n{vc.to_string()}")
print(f"Porcentaje de positivos (fallos): {pct_positivos:.2f}%")
print(f"Desbalanceo: {vc[0]/vc[1]:.1f}:1 (no-fallo:fallo)")


# ==============================================================
# PASO 2 — EDA
# ==============================================================
print("\n" + "=" * 60)
print("PASO 2 — EDA")
print("=" * 60)

# 2.1 Distribución del target — importante visualizar el desbalanceo
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels_pie = ["Sin Fallo (0)", "Con Fallo (1)"]
sizes = y.value_counts().sort_index().values
colors_pie = ["#2196F3", "#F44336"]
axes[0].pie(sizes, labels=labels_pie, colors=colors_pie, autopct="%1.1f%%",
            startangle=90, explode=(0, 0.05))
axes[0].set_title("Distribución del Target (Pie Chart)", fontweight="bold")
sns.countplot(x=y.astype(str), ax=axes[1],
              hue=y.astype(str),
              palette={"0": "#2196F3", "1": "#F44336"}, legend=False)
axes[1].set_title("Distribución del Target (Conteo)", fontweight="bold")
axes[1].set_xlabel("Machine Failure")
axes[1].set_ylabel("Cantidad de Registros")
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 f"{int(bar.get_height())}", ha="center", fontweight="bold")
plt.suptitle("Distribución de la Variable Objetivo — Machine Failure",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_01_distribucion_target.png")
plt.close()
print("Guardado: fig_01_distribucion_target.png")

# 2.2 Correlación — identificamos qué features se relacionan más con el fallo
numeric_features = numeric_cols + ["Machine failure"]
corr_df = df[numeric_features].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
ax.set_title("Matriz de Correlación — Features Numéricas + Target",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_02_correlacion.png")
plt.close()
print("Guardado: fig_02_correlacion.png")
print("Correlaciones con Machine failure:")
print(corr_df["Machine failure"].drop("Machine failure")
      .sort_values(key=abs, ascending=False).to_string())

# 2.3 Boxplots — distribución de cada feature separada por clase
df_plot = X[numeric_cols].copy()
df_plot["Fallo"] = y.map({0: "Sin Fallo", 1: "Con Fallo"})
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, feat in enumerate(numeric_cols):
    sns.boxplot(data=df_plot, x="Fallo", y=feat, ax=axes[i],
                palette={"Sin Fallo": "#2196F3", "Con Fallo": "#F44336"},
                hue="Fallo", legend=False)
    axes[i].set_title(feat, fontweight="bold", fontsize=9)
    axes[i].set_xlabel("")
axes[-1].set_visible(False)
plt.suptitle("Distribución de Features Numéricas por Clase",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_03_boxplots.png")
plt.close()
print("Guardado: fig_03_boxplots.png")

# 2.4 Pairplot — muestra de 500 por clase para evitar lentitud
df_pair = X[numeric_cols].copy()
df_pair["Fallo"] = y.map({0: "Sin Fallo", 1: "Con Fallo"})
sample_idx = (
    df_pair.groupby("Fallo")
    .apply(lambda g: g.sample(min(500, len(g)), random_state=RANDOM_STATE))
    .index.get_level_values(1)
)
g = sns.pairplot(df_pair.loc[sample_idx], hue="Fallo",
                 palette={"Sin Fallo": "#2196F3", "Con Fallo": "#F44336"},
                 plot_kws={"alpha": 0.5, "s": 15}, diag_kind="kde")
g.fig.suptitle("Pairplot — Features Numéricas (muestra 500 por clase)",
               y=1.02, fontsize=13, fontweight="bold")
plt.savefig(FIGS_DIR / "fig_04_pairplot.png")
plt.close()
print("Guardado: fig_04_pairplot.png")

# 2.5 Feature importance preliminar con un RF rápido (orientativo)
rf_prelim = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_prelim.fit(X, y)
importances = pd.Series(rf_prelim.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
colors_bar = ["#F44336" if v > importances.median() else "#2196F3" for v in importances]
importances.plot(kind="barh", ax=ax, color=colors_bar, edgecolor="white")
ax.set_title("Importancia de Features — Random Forest Preliminar",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importancia (Gini)")
for i, v in enumerate(importances):
    ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_05_feature_importance_rf.png")
plt.close()
print("Guardado: fig_05_feature_importance_rf.png")

# 2.6 Diferencia de temperatura — precursor de HDF identificado en Informe 1
df_temp = X.copy()
df_temp["temp_diff"] = df_temp["Process temperature [K]"] - df_temp["Air temperature [K]"]
df_temp["Fallo"] = y.values
fig, ax = plt.subplots(figsize=(10, 6))
for clase, color, lbl in [(0, "#2196F3", "Sin Fallo"), (1, "#F44336", "Con Fallo")]:
    ax.hist(df_temp[df_temp["Fallo"] == clase]["temp_diff"],
            bins=40, alpha=0.6, color=color, label=lbl, edgecolor="white")
ax.set_title("Diferencia de Temperatura (Proceso - Ambiente) por Clase",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Diferencia de Temperatura [K]")
ax.set_ylabel("Frecuencia")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_06_temp_diff.png")
plt.close()
print("Guardado: fig_06_temp_diff.png")
print(f"Diferencia media (Sin Fallo): {df_temp[df_temp['Fallo']==0]['temp_diff'].mean():.2f} K")
print(f"Diferencia media (Con Fallo): {df_temp[df_temp['Fallo']==1]['temp_diff'].mean():.2f} K")


# ==============================================================
# PASO 3 — Clustering
# ==============================================================
# Escalamos antes de cualquier algoritmo basado en distancias.
# PCA se usa solo para visualizar los clusters en 2D.
print("\n" + "=" * 60)
print("PASO 3 — Clustering")
print("=" * 60)

scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(X)

pca_vis = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca_vis.fit_transform(X_scaled)
print(f"Varianza explicada PCA 2D: {pca_vis.explained_variance_ratio_.sum():.3f}")
print(f"  PC1: {pca_vis.explained_variance_ratio_[0]:.3f}  |  PC2: {pca_vis.explained_variance_ratio_[1]:.3f}")

# ── 3.1 K-Means ──────────────────────────────────────────────
# Probamos k=2..10; elegimos el k con mayor Silhouette Score.
print("\n-- K-Means --")
inertias, silhouettes = [], []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, lbl, sample_size=2000, random_state=RANDOM_STATE)
    silhouettes.append(sil)
    print(f"  k={k}: inercia={km.inertia_:.0f}, silhouette={sil:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(list(k_range), inertias, "bo-", markersize=8, linewidth=2)
ax.set_title("K-Means: Método del Codo", fontsize=13, fontweight="bold")
ax.set_xlabel("Número de Clusters (k)")
ax.set_ylabel("Inercia (WCSS)")
ax.set_xticks(list(k_range))
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_07_kmeans_codo.png")
plt.close()
print("Guardado: fig_07_kmeans_codo.png")

best_k = list(k_range)[np.argmax(silhouettes)]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(list(k_range), silhouettes, "rs-", markersize=8, linewidth=2)
ax.axvline(x=best_k, color="green", linestyle="--", linewidth=2,
           label=f"k óptimo={best_k} (sil={max(silhouettes):.4f})")
ax.set_title("K-Means: Silhouette Score por k", fontsize=13, fontweight="bold")
ax.set_xlabel("Número de Clusters (k)")
ax.set_ylabel("Silhouette Score")
ax.set_xticks(list(k_range))
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_08_kmeans_silhouette.png")
plt.close()
print(f"Guardado: fig_08_kmeans_silhouette.png  |  k óptimo: {best_k}")

km_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
km_labels = km_best.fit_predict(X_scaled)
km_silhouette = silhouette_score(X_scaled, km_labels, sample_size=2000, random_state=RANDOM_STATE)
km_ari = adjusted_rand_score(y, km_labels)

fig, ax = plt.subplots(figsize=(10, 7))
palette_km = sns.color_palette("tab10", best_k)
for cid in range(best_k):
    mask = km_labels == cid
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[palette_km[cid]],
               label=f"Cluster {cid}", alpha=0.5, s=15, edgecolors="none")
ax.set_title(f"K-Means (k={best_k}) — Proyección PCA 2D\n"
             f"Silhouette={km_silhouette:.4f} | ARI={km_ari:.4f}",
             fontsize=12, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca_vis.explained_variance_ratio_[1]*100:.1f}% var)")
ax.legend(markerscale=2)
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_09_kmeans_pca.png")
plt.close()
print(f"Guardado: fig_09_kmeans_pca.png  |  Silhouette={km_silhouette:.4f} | ARI={km_ari:.4f}")

# ── 3.2 Fuzzy C-Means ────────────────────────────────────────
# Cada punto tiene membresía parcial en cada cluster (0..1).
# Elegimos c según el FPC (Fuzzy Partition Coefficient) más alto.
print("\n-- Fuzzy C-Means --")
try:
    X_fcm = X_scaled.T  # skfuzzy espera (n_features, n_samples)
    fpc_scores = {}
    for c in [2, 3, 4]:
        _, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            X_fcm, c, 2, error=0.005, maxiter=1000, seed=RANDOM_STATE
        )
        fpc_scores[c] = fpc
        print(f"  c={c}: FPC={fpc:.4f}")

    best_c = max(fpc_scores, key=fpc_scores.get)
    print(f"  Mejor c: {best_c} (FPC={fpc_scores[best_c]:.4f})")

    _, u_best, _, _, _, _, fpc_best = fuzz.cluster.cmeans(
        X_fcm, best_c, 2, error=0.005, maxiter=1000, seed=RANDOM_STATE
    )
    fcm_labels = np.argmax(u_best, axis=0)  # asignamos al cluster de mayor membresía
    fcm_silhouette = silhouette_score(X_scaled, fcm_labels, sample_size=2000,
                                      random_state=RANDOM_STATE)
    fcm_ari = adjusted_rand_score(y, fcm_labels)

    fig, ax = plt.subplots(figsize=(10, 7))
    palette_fcm = sns.color_palette("Set2", best_c)
    for cid in range(best_c):
        mask = fcm_labels == cid
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[palette_fcm[cid]],
                   label=f"Cluster {cid}", alpha=0.5, s=15, edgecolors="none")
    ax.set_title(f"Fuzzy C-Means (c={best_c}) — Proyección PCA 2D\n"
                 f"FPC={fpc_best:.4f} | Silhouette={fcm_silhouette:.4f} | ARI={fcm_ari:.4f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca_vis.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "fig_10_fuzzy_pca.png")
    plt.close()
    print(f"Guardado: fig_10_fuzzy_pca.png  |  Silhouette={fcm_silhouette:.4f} | ARI={fcm_ari:.4f}")

except Exception as e:
    print(f"ERROR Fuzzy C-Means: {e}")
    fcm_labels     = np.zeros(len(y), dtype=int)
    fcm_silhouette = 0.0
    fcm_ari        = 0.0
    best_c         = 1
    fpc_best       = 0.0

# ── 3.3 DBSCAN ───────────────────────────────────────────────
# Detecta clusters de forma arbitraria; marca puntos aislados como ruido (-1).
# El K-distance graph nos ayuda a estimar un buen valor de eps.
print("\n-- DBSCAN --")
try:
    nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_distances, linewidth=2)
    ax.set_title("DBSCAN: K-Distance Graph (k=5) — Estimación de epsilon",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Puntos ordenados (mayor a menor distancia)")
    ax.set_ylabel("Distancia al 5º vecino más cercano")
    ax.axhline(y=0.5, color="r", linestyle="--", label="eps≈0.5")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "fig_11_dbscan_kdist.png")
    plt.close()
    print("Guardado: fig_11_dbscan_kdist.png")

    # Grilla de hiperparámetros; elegimos la combinación con mejor Silhouette
    dbscan_results = []
    for eps in [0.3, 0.5, 0.8, 1.0]:
        for ms in [5, 10, 15]:
            lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_scaled)
            nc = len(set(lbl)) - (1 if -1 in lbl else 0)
            nn = (lbl == -1).sum()
            valid = lbl != -1
            if nc > 1 and valid.sum() > 10:
                sil = silhouette_score(X_scaled[valid], lbl[valid],
                                       sample_size=min(2000, valid.sum()),
                                       random_state=RANDOM_STATE)
            else:
                sil = np.nan
            dbscan_results.append({"eps": eps, "min_samples": ms,
                                   "n_clusters": nc, "n_noise": nn,
                                   "silhouette": round(sil, 4) if not np.isnan(sil) else np.nan})

    df_dbscan = pd.DataFrame(dbscan_results)
    print("Resultados DBSCAN:")
    print(df_dbscan.to_string(index=False))

    df_valid_dbs = df_dbscan.dropna(subset=["silhouette"])
    if len(df_valid_dbs) > 0:
        best_dbs = df_valid_dbs.loc[df_valid_dbs["silhouette"].idxmax()]
        eps_best, ms_best = best_dbs["eps"], int(best_dbs["min_samples"])
    else:
        eps_best, ms_best = 0.5, 10

    dbs_labels = DBSCAN(eps=eps_best, min_samples=ms_best).fit_predict(X_scaled)
    n_clusters_dbs = len(set(dbs_labels)) - (1 if -1 in dbs_labels else 0)
    dbs_ari = adjusted_rand_score(y, dbs_labels)
    valid_best = dbs_labels != -1
    if valid_best.sum() > 10 and n_clusters_dbs > 1:
        dbs_silhouette = silhouette_score(X_scaled[valid_best], dbs_labels[valid_best],
                                          sample_size=min(2000, valid_best.sum()),
                                          random_state=RANDOM_STATE)
    else:
        dbs_silhouette = 0.0

    fig, ax = plt.subplots(figsize=(10, 7))
    unique_dbs = sorted(set(dbs_labels))
    palette_dbs = sns.color_palette("tab10", len(unique_dbs))
    for i, lbl_id in enumerate(unique_dbs):
        mask = dbs_labels == lbl_id
        lbl_name = "Ruido" if lbl_id == -1 else f"Cluster {lbl_id}"
        color = "gray" if lbl_id == -1 else palette_dbs[i % len(palette_dbs)]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], label=lbl_name,
                   alpha=0.4 if lbl_id == -1 else 0.6,
                   s=10 if lbl_id == -1 else 15, edgecolors="none")
    ax.set_title(f"DBSCAN (eps={eps_best}, min_s={ms_best}) — Proyección PCA 2D\n"
                 f"Clusters={n_clusters_dbs} | Silhouette={dbs_silhouette:.4f} | ARI={dbs_ari:.4f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca_vis.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "fig_12_dbscan_pca.png")
    plt.close()
    print(f"Guardado: fig_12_dbscan_pca.png  |  Silhouette={dbs_silhouette:.4f} | ARI={dbs_ari:.4f}")

except Exception as e:
    print(f"ERROR DBSCAN: {e}")
    dbs_labels     = np.zeros(len(y), dtype=int)
    dbs_silhouette = 0.0
    dbs_ari        = 0.0
    n_clusters_dbs = 1

# ── 3.4 Mean Shift ───────────────────────────────────────────
# Aproximación de Subtractive Clustering: desplaza puntos hacia los modos
# de densidad sin necesitar k de antemano. bandwidth se estima automáticamente.
print("\n-- Mean Shift --")
try:
    bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=500,
                                   random_state=RANDOM_STATE)
    print(f"  Bandwidth estimado: {bandwidth:.4f}")
    ms_model  = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms_labels = ms_model.fit_predict(X_scaled)
    n_clusters_ms = len(np.unique(ms_labels))
    print(f"  Clusters encontrados: {n_clusters_ms}")
    ms_silhouette = (silhouette_score(X_scaled, ms_labels, sample_size=2000,
                                      random_state=RANDOM_STATE)
                     if n_clusters_ms > 1 else 0.0)
    ms_ari = adjusted_rand_score(y, ms_labels)

    fig, ax = plt.subplots(figsize=(10, 7))
    palette_ms = sns.color_palette("tab10", n_clusters_ms)
    for cid in range(n_clusters_ms):
        mask = ms_labels == cid
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[palette_ms[cid % len(palette_ms)]],
                   label=f"Cluster {cid}", alpha=0.5, s=15, edgecolors="none")
    ax.set_title(f"Mean Shift (bw={bandwidth:.3f}) — Proyección PCA 2D\n"
                 f"Clusters={n_clusters_ms} | Silhouette={ms_silhouette:.4f} | ARI={ms_ari:.4f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca_vis.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "fig_13_meanshift_pca.png")
    plt.close()
    print(f"Guardado: fig_13_meanshift_pca.png  |  Silhouette={ms_silhouette:.4f} | ARI={ms_ari:.4f}")

except Exception as e:
    print(f"ERROR Mean Shift: {e}")
    ms_labels     = np.zeros(len(y), dtype=int)
    ms_silhouette = 0.0
    ms_ari        = 0.0
    n_clusters_ms = 1

# ── 3.5 Tabla comparativa ────────────────────────────────────
clustering_comparison = pd.DataFrame([
    {"Método": "K-Means",
     "N_Clusters": best_k,
     "Silhouette_Score": round(km_silhouette, 4),
     "Adjusted_Rand_Index_vs_target": round(km_ari, 4)},
    {"Método": "Fuzzy C-Means",
     "N_Clusters": best_c,
     "Silhouette_Score": round(fcm_silhouette, 4),
     "Adjusted_Rand_Index_vs_target": round(fcm_ari, 4)},
    {"Método": "DBSCAN",
     "N_Clusters": n_clusters_dbs,
     "Silhouette_Score": round(dbs_silhouette, 4),
     "Adjusted_Rand_Index_vs_target": round(dbs_ari, 4)},
    {"Método": "Mean Shift",
     "N_Clusters": n_clusters_ms,
     "Silhouette_Score": round(ms_silhouette, 4),
     "Adjusted_Rand_Index_vs_target": round(ms_ari, 4)},
])
clustering_comparison.to_csv(RESULTS_DIR / "clustering_comparison.csv", index=False)
print("\nTabla comparativa de clustering:")
print(clustering_comparison.to_string(index=False))
print("Guardado: clustering_comparison.csv")


# ==============================================================
# PASO 4 — Reevaluación de Etiquetas
# ==============================================================
# Hasta el 30% de las etiquetas en datasets industriales pueden estar mal.
# Detectamos candidatas con dos fuentes independientes y solo corregimos
# las que ambas marcan (intersección conservadora).
print("\n" + "=" * 60)
print("PASO 4 — Reevaluación de Etiquetas")
print("=" * 60)

y_np = y.values

# 4.1 Consenso de clusters: sospechosa = etiqueta ≠ mayoría del cluster
cluster_majority = {}
for cid in range(best_k):
    mask = km_labels == cid
    majority = int(np.bincount(y_np[mask]).argmax())
    cluster_majority[cid] = majority
    print(f"  Cluster {cid}: etiqueta mayoritaria={majority} "
          f"(n={mask.sum()}, fallos_reales={y_np[mask].sum()})")

y_cluster_pred   = np.array([cluster_majority[lbl] for lbl in km_labels])
suspicious_cluster = set(np.where(y_np != y_cluster_pred)[0])
print(f"\nSospechosas (consenso clustering): {len(suspicious_cluster)} "
      f"({100*len(suspicious_cluster)/len(y_np):.2f}%)")

# 4.2 Confident Learning: y=1 con P(fallo)<0.20 o y=0 con P(fallo)>0.80
print("Calculando probabilidades OOF con Random Forest...")
rf_cl = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
proba_oof = cross_val_predict(
    rf_cl, X_scaled, y_np, cv=skf, method="predict_proba"
)[:, 1]

suspicious_model = set(np.where(
    ((y_np == 1) & (proba_oof < 0.20)) |
    ((y_np == 0) & (proba_oof > 0.80))
)[0])
print(f"Sospechosas (Confident Learning): {len(suspicious_model)} "
      f"({100*len(suspicious_model)/len(y_np):.2f}%)")
print(f"  y==1 pero P(fallo)<0.20: {((y_np==1)&(proba_oof<0.20)).sum()}")
print(f"  y==0 pero P(fallo)>0.80: {((y_np==0)&(proba_oof>0.80)).sum()}")

# 4.3 Intersección → flip de etiquetas → Dataset A (original) y B (corregido)
suspicious_both = suspicious_cluster & suspicious_model
n_changed   = len(suspicious_both)
pct_changed = 100 * n_changed / len(y_np)
print(f"\nSospechosas por clustering:   {len(suspicious_cluster)}")
print(f"Sospechosas por modelo:       {len(suspicious_model)}")
print(f"Intersección (flip):          {n_changed} ({pct_changed:.2f}%)")

y_A = y_np.copy()
y_B = y_np.copy()
for idx in suspicious_both:
    y_B[idx] = 1 - y_B[idx]

print(f"\nDataset A — distribución: {np.bincount(y_A)}")
print(f"Dataset B — distribución: {np.bincount(y_B)}")

df_A = X.copy(); df_A["Machine failure"] = y_A
df_B = X.copy(); df_B["Machine failure"] = y_B
df_A.to_csv(RESULTS_DIR / "dataset_A_original.csv",  index=False)
df_B.to_csv(RESULTS_DIR / "dataset_B_corregido.csv", index=False)
print("Guardado: dataset_A_original.csv y dataset_B_corregido.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_data, title, color in [
    (axes[0], y_A, "Dataset A — Original",  "#2196F3"),
    (axes[1], y_B, "Dataset B — Corregido", "#4CAF50"),
]:
    counts = np.bincount(y_data)
    bars = ax.bar(["Sin Fallo (0)", "Con Fallo (1)"], counts,
                  color=[color, "#F44336"], edgecolor="white", linewidth=1.5)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel("Cantidad de Registros")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{cnt}\n({100*cnt/len(y_data):.1f}%)", ha="center", fontsize=10)
plt.suptitle(f"Comparación de Distribución — Antes y Después de Corrección\n"
             f"({n_changed} etiquetas corregidas = {pct_changed:.2f}%)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_14_etiquetas_comparacion.png")
plt.close()
print("Guardado: fig_14_etiquetas_comparacion.png")


# ==============================================================
# PASO 5 — Modelos Supervisados
# ==============================================================
# Split 60/20/20: train / validación / test.
# Flujo: split → SMOTE en train → StandardScaler (fit en train).
# El set de validación se usa en GridSearchCV; el de test solo para métricas finales.
print("\n" + "=" * 60)
print("PASO 5 — Modelos Supervisados")
print("=" * 60)


def preparar_datos(X_data, y_data, random_state=42):
    """
    Split 60/20/20 estratificado → SMOTE en train → StandardScaler.
    Retorna dict con X_train, y_train, X_val, y_val, X_test, y_test, scaler.
    """
    # Primer split: 60% train, 40% temporal
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_data, y_data, test_size=0.40,
        stratify=y_data, random_state=random_state,
    )
    # Segundo split: 50% de temporal → 20% val y 20% test (sobre el total)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50,
        stratify=y_tmp, random_state=random_state,
    )
    # SMOTE solo en train para no contaminar val ni test
    X_tr_sm, y_tr_sm = SMOTE(random_state=random_state).fit_resample(X_tr, y_tr)
    # Scaler ajustado en train, aplicado a val y test
    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_sm)
    X_val_sc = scaler.transform(X_val)
    X_te_sc  = scaler.transform(X_te)
    return {
        "X_train": X_tr_sc,  "y_train": y_tr_sm,
        "X_val":   X_val_sc, "y_val":   y_val,
        "X_test":  X_te_sc,  "y_test":  y_te,
        "scaler":  scaler,
    }


datos_A = preparar_datos(X.values, y_A)
datos_B = preparar_datos(X.values, y_B)
print(f"Dataset A → Train: {datos_A['X_train'].shape} | Val: {datos_A['X_val'].shape} | Test: {datos_A['X_test'].shape}")
print(f"Dataset B → Train: {datos_B['X_train'].shape} | Val: {datos_B['X_val'].shape} | Test: {datos_B['X_test'].shape}")


def evaluar_modelo(model, X_test, y_test, nombre, dataset_name):
    """Calcula las 5 métricas estándar sobre el conjunto de test."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        dv = model.decision_function(X_test)
        y_proba = (dv - dv.min()) / (dv.max() - dv.min())
    else:
        y_proba = y_pred.astype(float)
    return {
        "Modelo":    nombre,
        "Dataset":   dataset_name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
        "_y_pred":   y_pred,
        "_y_proba":  y_proba,
    }


resultados         = []
modelos_entrenados = {}

MODELOS_NOMBRES = [
    "Árbol de Decisión",
    "Regresión Logística",
    "Regresión Lineal",
    "Random Forest",
    "SVM",
]

# ── 5.1 Árbol de Decisión ────────────────────────────────────
# Modelo interpretable; GridSearchCV sobre profundidad y criterio de split.
print("\n-- Árbol de Decisión --")
dt_params = {
    "max_depth":         [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "criterion":         ["gini", "entropy"],
}
for ds_name, datos in [("A", datos_A), ("B", datos_B)]:
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        dt_params, cv=5, scoring="f1", n_jobs=-1,
    )
    gs.fit(datos["X_train"], datos["y_train"])
    best = gs.best_estimator_
    modelos_entrenados[("Árbol de Decisión", ds_name)] = best
    res = evaluar_modelo(best, datos["X_test"], datos["y_test"],
                         "Árbol de Decisión", ds_name)
    resultados.append(res)
    print(f"  Dataset {ds_name}: F1={res['F1']:.4f} | AUC={res['ROC-AUC']:.4f} | "
          f"Recall={res['Recall']:.4f} | params={gs.best_params_}")

# Visualizamos el árbol con profundidad máxima 4 para que quepa en la imagen
best_dt_A = modelos_entrenados[("Árbol de Decisión", "A")]
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(best_dt_A, max_depth=4, feature_names=FEATURE_COLS,
          class_names=["Sin Fallo", "Con Fallo"],
          filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title("Árbol de Decisión — Dataset A (profundidad visual=4)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_15_arbol_decision.png")
plt.close()
print("Guardado: fig_15_arbol_decision.png")

fi_dt = pd.Series(best_dt_A.feature_importances_, index=FEATURE_COLS).sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
fi_dt.plot(kind="barh", ax=ax, color="#FF7043", edgecolor="white")
ax.set_title("Importancia de Features — Árbol de Decisión", fontsize=13, fontweight="bold")
ax.set_xlabel("Importancia Gini")
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_16_arbol_importancia.png")
plt.close()
print("Guardado: fig_16_arbol_importancia.png")

# ── 5.2 Regresión Logística ──────────────────────────────────
# Modelo probabilístico lineal; coeficientes indican influencia de cada feature.
print("\n-- Regresión Logística --")
lr_params = {
    "C":        [0.01, 0.1, 1, 10, 100],
    "solver":   ["lbfgs", "liblinear"],
    "max_iter": [1000],
}
for ds_name, datos in [("A", datos_A), ("B", datos_B)]:
    gs = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE),
        lr_params, cv=5, scoring="f1", n_jobs=-1,
    )
    gs.fit(datos["X_train"], datos["y_train"])
    best = gs.best_estimator_
    modelos_entrenados[("Regresión Logística", ds_name)] = best
    res = evaluar_modelo(best, datos["X_test"], datos["y_test"],
                         "Regresión Logística", ds_name)
    resultados.append(res)
    print(f"  Dataset {ds_name}: F1={res['F1']:.4f} | AUC={res['ROC-AUC']:.4f} | "
          f"Recall={res['Recall']:.4f} | params={gs.best_params_}")

best_lr_A = modelos_entrenados[("Regresión Logística", "A")]
coefs = pd.Series(best_lr_A.coef_[0], index=FEATURE_COLS).sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
colors_coef = ["#F44336" if c > 0 else "#2196F3" for c in coefs]
coefs.plot(kind="barh", ax=ax, color=colors_coef, edgecolor="white")
ax.axvline(x=0, color="black", linewidth=1)
ax.set_title("Coeficientes — Regresión Logística (Dataset A)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Coeficiente (positivo = mayor prob. de fallo)")
red_p  = mpatches.Patch(color="#F44336", label="Aumenta prob. fallo")
blue_p = mpatches.Patch(color="#2196F3", label="Reduce prob. fallo")
ax.legend(handles=[red_p, blue_p])
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_17_logistica_coeficientes.png")
plt.close()
print("Guardado: fig_17_logistica_coeficientes.png")

# ── 5.3 Regresión Lineal (baseline) ──────────────────────────
# No acota la salida en [0,1]; se incluye como piso comparativo con umbral=0.5.
print("\n-- Regresión Lineal (baseline) --")
for ds_name, datos in [("A", datos_A), ("B", datos_B)]:
    linreg = LinearRegression()
    linreg.fit(datos["X_train"], datos["y_train"])
    modelos_entrenados[("Regresión Lineal", ds_name)] = linreg
    y_raw  = linreg.predict(datos["X_test"])
    y_pred = (y_raw >= 0.5).astype(int)
    y_prob = np.clip(y_raw, 0, 1)  # clipeamos para poder calcular AUC
    res = {
        "Modelo":    "Regresión Lineal",
        "Dataset":   ds_name,
        "Accuracy":  round(accuracy_score(datos["y_test"], y_pred), 4),
        "Precision": round(precision_score(datos["y_test"], y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(datos["y_test"], y_pred, zero_division=0), 4),
        "F1":        round(f1_score(datos["y_test"], y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(datos["y_test"], y_prob), 4),
        "_y_pred":   y_pred,
        "_y_proba":  y_prob,
    }
    resultados.append(res)
    print(f"  Dataset {ds_name}: F1={res['F1']:.4f} | AUC={res['ROC-AUC']:.4f} | Recall={res['Recall']:.4f}")

# ── 5.4 Random Forest ────────────────────────────────────────
# Ensamble de 300 árboles; si hay cuML (GPU), lo usa automáticamente.
print("\n-- Random Forest (300 árboles) --")
use_cuml_rf = False
if device.type == "cuda":
    try:
        from cuml.ensemble import RandomForestClassifier as cuRF
        use_cuml_rf = True
        print("  cuML RandomForest disponible — usando GPU")
    except ImportError:
        print("  cuML no disponible — usando sklearn CPU")

for ds_name, datos in [("A", datos_A), ("B", datos_B)]:
    rf = (cuRF if use_cuml_rf else RandomForestClassifier)(
        n_estimators=300, random_state=RANDOM_STATE,
        **({"n_jobs": -1} if not use_cuml_rf else {}),
    )
    rf.fit(datos["X_train"], datos["y_train"])
    modelos_entrenados[("Random Forest", ds_name)] = rf
    res = evaluar_modelo(rf, datos["X_test"], datos["y_test"], "Random Forest", ds_name)
    resultados.append(res)
    print(f"  Dataset {ds_name}: F1={res['F1']:.4f} | AUC={res['ROC-AUC']:.4f} | Recall={res['Recall']:.4f}")

# ── 5.5 SVM ──────────────────────────────────────────────────
# Kernel RBF; GridSearchCV sobre C (regularización) y gamma (ancho del kernel).
print("\n-- SVM --")
svm_params = {
    "C":     [0.1, 1, 10],
    "gamma": ["scale", "auto"],
}
for ds_name, datos in [("A", datos_A), ("B", datos_B)]:
    gs = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        svm_params, cv=5, scoring="f1", n_jobs=-1,
    )
    gs.fit(datos["X_train"], datos["y_train"])
    best = gs.best_estimator_
    modelos_entrenados[("SVM", ds_name)] = best
    res = evaluar_modelo(best, datos["X_test"], datos["y_test"], "SVM", ds_name)
    resultados.append(res)
    print(f"  Dataset {ds_name}: F1={res['F1']:.4f} | AUC={res['ROC-AUC']:.4f} | "
          f"Recall={res['Recall']:.4f} | params={gs.best_params_}")

# ── 5.6 Matrices de confusión ────────────────────────────────
# FN (fallo no detectado) es el error más costoso en contexto industrial.
def plot_confusion_matrices(dataset_name, datos, fig_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, model_name in enumerate(MODELOS_NOMBRES):
        key = (model_name, dataset_name)
        if key not in modelos_entrenados:
            axes[i].set_visible(False)
            continue
        model = modelos_entrenados[key]
        if model_name == "Regresión Lineal":
            y_pred = (model.predict(datos["X_test"]) >= 0.5).astype(int)
        else:
            y_pred = model.predict(datos["X_test"])
        cm = confusion_matrix(datos["y_test"], y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Sin Fallo", "Con Fallo"],
                    yticklabels=["Sin Fallo", "Con Fallo"])
        axes[i].set_title(model_name, fontweight="bold", fontsize=11)
        axes[i].set_xlabel("Predicho")
        axes[i].set_ylabel("Real")
    axes[-1].set_visible(False)
    plt.suptitle(f"Matrices de Confusión — Dataset {dataset_name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

plot_confusion_matrices("A", datos_A, FIGS_DIR / "fig_18_confusion_matrices_A.png")
print("Guardado: fig_18_confusion_matrices_A.png")
plot_confusion_matrices("B", datos_B, FIGS_DIR / "fig_19_confusion_matrices_B.png")
print("Guardado: fig_19_confusion_matrices_B.png")


# ==============================================================
# PASO 6 — Comparativa Final
# ==============================================================
print("\n" + "=" * 60)
print("PASO 6 — Comparativa Final")
print("=" * 60)

# 6.1 Tabla maestra ordenada por F1
metrics_keys = ["Modelo", "Dataset", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
df_results = pd.DataFrame([{k: r[k] for k in metrics_keys} for r in resultados])
df_results = df_results.sort_values("F1", ascending=False).reset_index(drop=True)
df_results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
print("Tabla maestra (ordenada por F1):")
print(df_results.to_string(index=False))
print("Guardado: model_comparison.csv")

# 6.2 Grouped bar charts — Dataset A vs B para F1 y AUC
df_pivot_f1  = df_results.pivot(index="Modelo", columns="Dataset", values="F1")
df_pivot_auc = df_results.pivot(index="Modelo", columns="Dataset", values="ROC-AUC")

fig, ax = plt.subplots(figsize=(12, 6))
df_pivot_f1.plot(kind="bar", ax=ax, color=["#2196F3", "#4CAF50"],
                 edgecolor="white", width=0.7)
ax.set_title("Comparación F1-Score: Dataset A vs B por Modelo",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Modelo")
ax.set_ylabel("F1-Score")
ax.set_ylim(0, 1.05)
ax.legend(title="Dataset", labels=["Dataset A (original)", "Dataset B (corregido)"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_20_f1_comparacion.png")
plt.close()
print("Guardado: fig_20_f1_comparacion.png")

fig, ax = plt.subplots(figsize=(12, 6))
df_pivot_auc.plot(kind="bar", ax=ax, color=["#FF7043", "#AB47BC"],
                  edgecolor="white", width=0.7)
ax.set_title("Comparación ROC-AUC: Dataset A vs B por Modelo",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Modelo")
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0, 1.05)
ax.legend(title="Dataset", labels=["Dataset A (original)", "Dataset B (corregido)"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_21_auc_comparacion.png")
plt.close()
print("Guardado: fig_21_auc_comparacion.png")

# Heatmap de todas las métricas normalizadas
metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
df_hm = df_results.copy()
df_hm["Modelo_Dataset"] = df_hm["Modelo"] + " (" + df_hm["Dataset"] + ")"
df_hm = df_hm.set_index("Modelo_Dataset")[metric_cols]
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df_hm, annot=True, fmt=".3f", cmap="YlOrRd",
            vmin=0, vmax=1, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Heatmap de Métricas — Todos los Modelos y Datasets",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS_DIR / "fig_22_heatmap_metricas.png")
plt.close()
print("Guardado: fig_22_heatmap_metricas.png")

# 6.3 Curvas ROC superpuestas por dataset
def plot_roc_curves(dataset_name, datos, fig_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_roc = ["#F44336", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, model_name in enumerate(MODELOS_NOMBRES):
        key = (model_name, dataset_name)
        if key not in modelos_entrenados:
            continue
        model = modelos_entrenados[key]
        if model_name == "Regresión Lineal":
            y_proba = np.clip(model.predict(datos["X_test"]), 0, 1)
        elif hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(datos["X_test"])[:, 1]
        elif hasattr(model, "decision_function"):
            dv = model.decision_function(datos["X_test"])
            y_proba = (dv - dv.min()) / (dv.max() - dv.min())
        else:
            continue
        auc_val = roc_auc_score(datos["y_test"], y_proba)
        fpr, tpr, _ = roc_curve(datos["y_test"], y_proba)
        ax.plot(fpr, tpr, color=colors_roc[i % len(colors_roc)], linewidth=2,
                label=f"{model_name} (AUC={auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Clasificador aleatorio")
    ax.set_title(f"Curvas ROC — Dataset {dataset_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR = Recall)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

plot_roc_curves("A", datos_A, FIGS_DIR / "fig_23_roc_dataset_A.png")
print("Guardado: fig_23_roc_dataset_A.png")
plot_roc_curves("B", datos_B, FIGS_DIR / "fig_24_roc_dataset_B.png")
print("Guardado: fig_24_roc_dataset_B.png")


# ==============================================================
# RESUMEN FINAL
# ==============================================================
print("\n" + "=" * 60)
print("VALORES CLAVE PARA PRESENTACION HTML")
print("=" * 60)
print(f"Total registros: {len(y_np)}")
print(f"Fallos (y=1):    {y_np.sum()} ({100*y_np.mean():.2f}%)")
print(f"Sin fallos (y=0):{(y_np==0).sum()} ({100*(1-y_np.mean()):.2f}%)")
print(f"Etiquetas corregidas: {n_changed} ({pct_changed:.2f}%)")
print("\nClustering:")
print(clustering_comparison.to_string(index=False))
print("\nTop 5 modelos por F1:")
print(df_results[["Modelo", "Dataset", "F1", "ROC-AUC", "Recall"]].head(5).to_string(index=False))
best_row = df_results.iloc[0]
print(f"\nMejor: {best_row['Modelo']} (Dataset {best_row['Dataset']}) "
      f"F1={best_row['F1']:.4f} | AUC={best_row['ROC-AUC']:.4f} | Recall={best_row['Recall']:.4f}")

print("\n" + "=" * 60)
print("ARCHIVOS GENERADOS")
print("=" * 60)
print("\nFiguras (outputs/figures/):")
for f in sorted(FIGS_DIR.glob("*.png")):
    print(f"  [OK] {f.name} ({f.stat().st_size/1024:.1f} KB)")
print("\nResultados (outputs/results/):")
for f in sorted(RESULTS_DIR.glob("*")):
    print(f"  [OK] {f.name} ({f.stat().st_size/1024:.1f} KB)")
pres_dir = BASE_DIR / "presentacion"
print("\nPresentacion (presentacion/):")
for f in sorted(pres_dir.glob("*")):
    print(f"  [OK] {f.name} ({f.stat().st_size/1024:.1f} KB)")

print("\n" + "=" * 60)
print("Pipeline completado exitosamente.")
print("=" * 60)
