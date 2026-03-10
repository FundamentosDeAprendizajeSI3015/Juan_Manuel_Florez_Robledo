"""
FIRE-UdeA — Fase 5: Entrenamiento
===================================
Cuatro modelos con regularización:
1. Logistic Regression (L2, C=0.05)
2. Decision Tree (max_depth=3, min_samples_leaf=8)
3. Random Forest (max_depth=2, min_samples_leaf=8)
4. Gradient Boosting regularizado (n=30, depth=2, lr=0.03)

Los mismos modelos se aplican a ambos datasets.
Con más datos (500 filas) los árboles pueden generalizar mejor.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    brier_score_loss, confusion_matrix,
)

SEED = 42


def definir_modelos() -> dict:
    """Define los 4 modelos candidatos."""
    return {
        "LogisticRegression": LogisticRegression(
            C=0.05, l1_ratio=0, solver="lbfgs", max_iter=1000, random_state=SEED,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=8, random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=2, min_samples_leaf=8,
            max_features=5, max_samples=0.7, random_state=SEED,
        ),
        "GradientBoosting_reg": GradientBoostingClassifier(
            n_estimators=30, max_depth=2, learning_rate=0.03,
            min_samples_leaf=8, subsample=0.7, max_features=5,
            random_state=SEED,
        ),
    }


def evaluar_modelo(model, X, y_true):
    """Calcula métricas para un modelo en un split."""
    prob = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "n": len(y_true),
        "roc_auc": roc_auc_score(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def entrenar_y_comparar(data: dict) -> dict:
    """Entrena todos los modelos, compara y selecciona el mejor."""
    print("=" * 60)
    print("  Fase 5 — Entrenamiento y Selección de Modelo")
    print("=" * 60)

    models = definir_modelos()
    trained_models = {}
    results = []

    for name, model in models.items():
        print(f"\n  Entrenando {name}...")
        model.fit(data["X_train_s"], data["y_train"])
        trained_models[name] = model

        for split_name, X_s, y_true in [
            ("train", data["X_train_s"], data["y_train"]),
            ("val", data["X_val_s"], data["y_val"]),
            ("test", data["X_test_s"], data["y_test"]),
        ]:
            metrics = evaluar_modelo(model, X_s, y_true)
            metrics["modelo"] = name
            metrics["split"] = split_name
            results.append(metrics)

        train_auc = [r for r in results if r["modelo"] == name and r["split"] == "train"][0]["roc_auc"]
        val_auc = [r for r in results if r["modelo"] == name and r["split"] == "val"][0]["roc_auc"]
        test_auc = [r for r in results if r["modelo"] == name and r["split"] == "test"][0]["roc_auc"]
        gap = train_auc - test_auc
        emoji = "✅" if gap < 0.15 else "⚠️" if gap < 0.30 else "❌"
        print(f"    {emoji} AUC → Train: {train_auc:.3f}, Val: {val_auc:.3f}, Test: {test_auc:.3f} (gap: {gap:.3f})")

    # Seleccionar mejor modelo
    val_results = [r for r in results if r["split"] == "val"]
    for r in val_results:
        r["score"] = r["roc_auc"] * 0.5 + r["f1"] * 0.5
    best = max(val_results, key=lambda r: r["score"])
    best_name = best["modelo"]
    best_model = trained_models[best_name]

    print(f"\n  🏆 Mejor modelo: {best_name} (score val = {best['score']:.3f})")

    # Optimizar threshold
    prob_val = best_model.predict_proba(data["X_val_s"])[:, 1]
    thresholds = np.arange(0.20, 0.70, 0.01)
    f1_scores = [
        f1_score(data["y_val"], (prob_val >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"  Threshold óptimo: {best_threshold:.2f} (F1 val = {max(f1_scores):.3f})")
    print()

    return {
        "trained_models": trained_models,
        "results": results,
        "best_name": best_name,
        "best_model": best_model,
        "best_threshold": best_threshold,
    }