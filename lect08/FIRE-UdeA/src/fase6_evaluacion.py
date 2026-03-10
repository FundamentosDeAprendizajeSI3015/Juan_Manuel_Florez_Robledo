"""
FIRE-UdeA — Fase 6: Evaluación
================================
- Classification Report en test
- Comparación vs baseline (solo dataset realista)
- Curva ROC y Matriz de confusión
- Comparación de modelos por split
- Análisis de overfitting
- Feature importance
- Scores detallados
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, brier_score_loss, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve,
)

# Métricas del baseline proporcionado por la profesora (solo aplica a realista)
BASELINE = {
    "roc_auc": 0.4167, "f1": 0.857, "brier": 0.257,
    "accuracy": 0.25, "precision": 0.75, "recall": 1.00,
    "train_auc": 1.00, "gap": 0.583,
}


def evaluar_test(data: dict, training: dict) -> dict:
    """Evaluación final del mejor modelo en test."""
    es_realista = data.get("es_realista", False)

    print("=" * 60)
    print("  Fase 6 — Evaluación Final")
    print("=" * 60)

    best_model = training["best_model"]
    threshold = training["best_threshold"]

    prob_test = best_model.predict_proba(data["X_test_s"])[:, 1]
    pred_test = (prob_test >= threshold).astype(int)
    y_test = data["y_test"]

    new_auc = roc_auc_score(y_test, prob_test)
    new_f1 = f1_score(y_test, pred_test)
    new_brier = brier_score_loss(y_test, prob_test)
    new_acc = (y_test.values == pred_test).mean()
    new_prec = precision_score(y_test, pred_test, zero_division=0)
    new_rec = recall_score(y_test, pred_test, zero_division=0)

    if es_realista:
        print(f"\n  {'Métrica':<15} {'Baseline':>10} {'Nuevo':>10} {'Δ':>10}")
        print(f"  {'-'*45}")
        print(f"  {'ROC-AUC':<15} {BASELINE['roc_auc']:>10.4f} {new_auc:>10.4f} {new_auc - BASELINE['roc_auc']:>+10.4f}")
        print(f"  {'F1-Score':<15} {BASELINE['f1']:>10.4f} {new_f1:>10.4f} {new_f1 - BASELINE['f1']:>+10.4f}")
        print(f"  {'Brier':<15} {BASELINE['brier']:>10.4f} {new_brier:>10.4f} {new_brier - BASELINE['brier']:>+10.4f}")
        print(f"  {'Accuracy':<15} {BASELINE['accuracy']:>10.4f} {new_acc:>10.4f} {new_acc - BASELINE['accuracy']:>+10.4f}")
    else:
        print(f"\n  {'Métrica':<15} {'Valor':>10}")
        print(f"  {'-'*25}")
        print(f"  {'ROC-AUC':<15} {new_auc:>10.4f}")
        print(f"  {'F1-Score':<15} {new_f1:>10.4f}")
        print(f"  {'Brier':<15} {new_brier:>10.4f}")
        print(f"  {'Accuracy':<15} {new_acc:>10.4f}")
        print(f"  {'Precision':<15} {new_prec:>10.4f}")
        print(f"  {'Recall':<15} {new_rec:>10.4f}")

    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, pred_test, target_names=["Sano (0)", "Tensión (1)"]))

    return {
        "prob_test": prob_test, "pred_test": pred_test,
        "auc": new_auc, "f1": new_f1, "brier": new_brier,
        "accuracy": new_acc, "precision": new_prec, "recall": new_rec,
    }


def generar_graficas(data: dict, training: dict, evaluacion: dict, output_dir: str = "outputs"):
    """Genera todas las gráficas de evaluación."""
    os.makedirs(output_dir, exist_ok=True)

    y_test = data["y_test"]
    prob_test = evaluacion["prob_test"]
    pred_test = evaluacion["pred_test"]
    best_name = training["best_name"]
    feature_cols = data["feature_cols"]
    best_model = training["best_model"]
    results = training["results"]
    es_realista = data.get("es_realista", False)
    prefijo = "realista" if es_realista else "sintetico"

    # ── 1. Comparación vs Baseline (solo realista) ──
    if es_realista:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Comparación: Baseline (Profesora) vs Modelo Propio", fontsize=14, fontweight="bold")

        metricas = ["ROC-AUC", "F1-Score", "Accuracy", "Precision", "Recall"]
        vals_baseline = [BASELINE["roc_auc"], BASELINE["f1"], BASELINE["accuracy"],
                         BASELINE["precision"], BASELINE["recall"]]
        vals_nuevo = [evaluacion["auc"], evaluacion["f1"], evaluacion["accuracy"],
                      evaluacion["precision"], evaluacion["recall"]]

        x = np.arange(len(metricas))
        width = 0.35
        bars1 = axes[0].bar(x - width/2, vals_baseline, width, label="Baseline (Profe)",
                            color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=0.5)
        bars2 = axes[0].bar(x + width/2, vals_nuevo, width, label="Modelo Propio",
                            color="#2ecc71", alpha=0.8, edgecolor="black", linewidth=0.5)
        for bar in bars1:
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                         f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                         f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        axes[0].set_ylim(0, 1.15)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metricas, rotation=20, ha="right")
        axes[0].set_ylabel("Score")
        axes[0].set_title("Métricas en Test")
        axes[0].legend()

        brier_vals = [BASELINE["brier"], evaluacion["brier"]]
        colors_b = ["#e74c3c", "#2ecc71"]
        bars = axes[1].bar(["Baseline", "Modelo Propio"], brier_vals, color=colors_b,
                           alpha=0.8, edgecolor="black", linewidth=0.5, width=0.5)
        for bar, val in zip(bars, brier_vals):
            axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        axes[1].set_ylabel("Brier Score (↓ mejor)")
        axes[1].set_title("Calibración de Probabilidades")
        axes[1].set_ylim(0, 0.35)

        train_res = [r for r in results if r["modelo"] == best_name and r["split"] == "train"][0]
        real_gap = train_res["roc_auc"] - evaluacion["auc"]
        gap_data = [BASELINE["gap"], real_gap]
        bars = axes[2].bar(["Baseline", "Modelo Propio"], gap_data, color=colors_b,
                           alpha=0.8, edgecolor="black", linewidth=0.5, width=0.5)
        for bar, val in zip(bars, gap_data):
            axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        axes[2].axhline(y=0.15, color="orange", linestyle="--", alpha=0.7, label="Umbral aceptable (0.15)")
        axes[2].set_ylabel("Gap Train - Test AUC")
        axes[2].set_title("Overfitting (↓ mejor)")
        axes[2].set_ylim(0, 0.7)
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefijo}_comparacion_vs_baseline.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Guardado: {output_dir}/{prefijo}_comparacion_vs_baseline.png")

    # ── 2. ROC + Matriz de Confusión ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titulo = "Dataset Realista" if es_realista else "Dataset Sintético (500)"
    fig.suptitle(f"Evaluación — {titulo}", fontsize=13, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_test, prob_test)
    axes[0].plot(fpr, tpr, "b-", linewidth=2, label=f"{best_name} (AUC={evaluacion['auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Aleatorio (AUC=0.500)")
    if es_realista:
        axes[0].axhline(y=BASELINE["roc_auc"], color="red", linestyle=":", alpha=0.5,
                        label=f"Baseline AUC={BASELINE['roc_auc']:.3f}")
    axes[0].fill_between(fpr, tpr, alpha=0.1)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("Curva ROC — Test")
    axes[0].legend(fontsize=9)

    cm = confusion_matrix(y_test, pred_test)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sano", "Tensión"],
                yticklabels=["Sano", "Tensión"], ax=axes[1])
    axes[1].set_xlabel("Predicho")
    axes[1].set_ylabel("Real")
    axes[1].set_title("Matriz de Confusión — Test")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefijo}_roc_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Guardado: {output_dir}/{prefijo}_roc_confusion.png")

    # ── 3. Comparación de modelos por split ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Comparación de Modelos — {titulo}", fontsize=13, fontweight="bold")
    results_df = pd.DataFrame(results)

    for i, metric in enumerate(["roc_auc", "f1"]):
        pivot = results_df.pivot_table(index="modelo", columns="split", values=metric)
        pivot = pivot[["train", "val", "test"]]
        pivot.plot(kind="bar", ax=axes[i], rot=15)
        axes[i].set_title(f"{metric.upper()} por split")
        axes[i].set_ylim(0, 1.05)
        if es_realista:
            axes[i].axhline(y=BASELINE["roc_auc"], color="red", linestyle="--",
                            alpha=0.5, label=f"Baseline AUC={BASELINE['roc_auc']:.3f}")
        axes[i].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefijo}_comparacion_modelos.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Guardado: {output_dir}/{prefijo}_comparacion_modelos.png")

    # ── 4. Análisis de Overfitting ──
    fig, ax = plt.subplots(figsize=(10, 5))

    nombres = []
    train_aucs = []
    test_aucs = []
    gaps = []

    if es_realista:
        nombres.append("Baseline\n(Profesora)")
        train_aucs.append(BASELINE["train_auc"])
        test_aucs.append(BASELINE["roc_auc"])
        gaps.append(BASELINE["gap"])

    for name in results_df["modelo"].unique():
        tr = results_df[(results_df["modelo"] == name) & (results_df["split"] == "train")]["roc_auc"].values[0]
        te = results_df[(results_df["modelo"] == name) & (results_df["split"] == "test")]["roc_auc"].values[0]
        nombres.append(name.replace("_", "\n"))
        train_aucs.append(tr)
        test_aucs.append(te)
        gaps.append(tr - te)

    x = np.arange(len(nombres))
    width = 0.35
    ax.bar(x - width/2, train_aucs, width, label="Train AUC", color="#3498db", alpha=0.8)
    ax.bar(x + width/2, test_aucs, width, label="Test AUC", color="#e67e22", alpha=0.8)

    for i, gap in enumerate(gaps):
        color = "#27ae60" if gap < 0.15 else "#e67e22" if gap < 0.30 else "#e74c3c"
        ax.text(i, max(train_aucs[i], test_aucs[i]) + 0.03,
                f"gap: {gap:.3f}", ha="center", fontsize=8, fontweight="bold", color=color)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, label="Aleatorio")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=8)
    ax.set_ylabel("AUC")
    ax.set_title(f"Análisis de Overfitting — {titulo}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefijo}_overfitting.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Guardado: {output_dir}/{prefijo}_overfitting.png")

    # ── 5. Feature Importance ──
    if hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
        method = "Coeficientes absolutos (LogReg)"
    elif hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        method = "Feature importances (tree-based)"
    else:
        importances = np.zeros(len(feature_cols))
        method = "No disponible"

    imp_df = pd.DataFrame({
        "feature": feature_cols, "importance": importances,
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, max(6, len(feature_cols) * 0.3)))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(imp_df)))
    plt.barh(imp_df["feature"], imp_df["importance"], color=colors)
    plt.xlabel(f"Importancia ({method})")
    plt.title(f"Feature Importance — {best_name} ({titulo})")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefijo}_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Guardado: {output_dir}/{prefijo}_feature_importance.png")

    # ── 6. Visualización del Árbol de Decisión ──
    from sklearn.tree import plot_tree, DecisionTreeClassifier

    trained_models = training["trained_models"]

    # Si hay DecisionTree, graficar ese directamente
    if "DecisionTree" in trained_models:
        tree_model = trained_models["DecisionTree"]
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            tree_model,
            feature_names=feature_cols,
            class_names=["Sano", "Tensión"],
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax,
            proportion=True,
        )
        ax.set_title(f"Árbol de Decisión (max_depth=3) — {titulo}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefijo}_arbol_decision.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Guardado: {output_dir}/{prefijo}_arbol_decision.png")

    # También graficar un árbol del RandomForest (el primero del ensemble)
    if "RandomForest" in trained_models:
        rf_model = trained_models["RandomForest"]
        single_tree = rf_model.estimators_[0]
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(
            single_tree,
            feature_names=feature_cols,
            class_names=["Sano", "Tensión"],
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax,
            proportion=True,
        )
        ax.set_title(f"Árbol individual del RandomForest (1 de 200) — {titulo}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefijo}_arbol_random_forest.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Guardado: {output_dir}/{prefijo}_arbol_random_forest.png")


def scores_por_unidad(data: dict, evaluacion: dict) -> pd.DataFrame:
    """Genera tabla de scores detallados en test."""
    es_realista = data.get("es_realista", False)

    if es_realista:
        scores = data["test_df"][["anio", "unidad"]].copy()
    else:
        scores = pd.DataFrame({"indice": data["test_df"].index})

    scores["y_true"] = data["y_test"].values
    scores["probabilidad"] = evaluacion["prob_test"].round(4)
    scores["prediccion"] = evaluacion["pred_test"]
    scores["acierto"] = np.where(scores["y_true"] == scores["prediccion"], "✅", "❌")

    accuracy = (scores["y_true"] == scores["prediccion"]).mean()
    print(f"\n  Accuracy test: {accuracy:.1%}")

    if es_realista:
        print(f"\n  Scores por unidad:")
        print(scores.to_string(index=False))
    else:
        aciertos = scores["acierto"].value_counts()
        print(f"  Correctos: {aciertos.get('✅', 0)} / {len(scores)}")

    print()
    return scores