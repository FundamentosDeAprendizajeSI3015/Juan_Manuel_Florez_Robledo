"""
FIRE-UdeA — Fase 4: Espacio de Características y División
==========================================================
Adaptado según el dataset:

Dataset Realista: Split temporal (Train ≤2022, Val 2023, Test 2024-25)
Dataset Sintético: Split estratificado aleatorio (60/20/20)

En ambos: estandarización con fit solo en train.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 42


def separar_features_target(df: pd.DataFrame) -> dict:
    """
    Separa features, target y realiza split adaptado al dataset.

    Returns:
        dict con X_train_s, X_val_s, X_test_s, y_train, y_val, y_test,
        feature_cols, scaler, y DataFrames originales.
    """
    es_realista = "unidad" in df.columns

    print("=" * 60)
    tipo = "Temporal" if es_realista else "Estratificado Aleatorio"
    print(f"  Fase 4 — Espacio de Características y Split ({tipo})")
    print("=" * 60)

    # Columnas a excluir
    if es_realista:
        drop_cols = ["anio", "unidad", "label", "ingresos_totales", "gastos_personal"]
    else:
        drop_cols = ["label"]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    if es_realista:
        # ── Split temporal ──
        train = df[df["anio"] <= 2022].copy()
        val = df[df["anio"] == 2023].copy()
        test = df[df["anio"] >= 2024].copy()
        print(f"\n  ⚠️ Split temporal (evita data leakage):")
    else:
        # ── Split estratificado 60/20/20 ──
        train_val, test = train_test_split(
            df, test_size=0.20, stratify=df["label"], random_state=SEED
        )
        train, val = train_test_split(
            train_val, test_size=0.25, stratify=train_val["label"], random_state=SEED
        )
        print(f"\n  Split estratificado (60/20/20 con stratify):")

    X_train, y_train = train[feature_cols], train["label"]
    X_val, y_val = val[feature_cols], val["label"]
    X_test, y_test = test[feature_cols], test["label"]

    # Estandarizar
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print(f"  Features seleccionados: {len(feature_cols)}")
    print(f"    Train:  {X_train.shape[0]:4d} filas — prevalencia: {y_train.mean():.3f}")
    print(f"    Val:    {X_val.shape[0]:4d} filas — prevalencia: {y_val.mean():.3f}")
    print(f"    Test:   {X_test.shape[0]:4d} filas — prevalencia: {y_test.mean():.3f}")
    print(f"\n  Estandarización: fit en train, transform en val/test")
    print()

    return {
        "X_train_s": X_train_s, "X_val_s": X_val_s, "X_test_s": X_test_s,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_cols": feature_cols, "scaler": scaler,
        "train_df": train, "val_df": val, "test_df": test,
        "es_realista": es_realista,
    }