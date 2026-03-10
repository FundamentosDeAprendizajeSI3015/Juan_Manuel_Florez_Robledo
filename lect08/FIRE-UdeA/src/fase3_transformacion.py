"""
FIRE-UdeA — Fase 3: Transformación y Feature Engineering
=========================================================
Adaptado según el dataset:

Dataset Realista (panel temporal):
  - Lag features (t-1), diferencias interanuales, rolling mean

Dataset Sintético (500 filas, sin temporalidad):
  - Interacciones entre features, flags binarios, ratios

Común a ambos:
  - Flag CFO negativo
  - Interacciones clave
"""
import os
import pandas as pd
import numpy as np


# ── Features para dataset REALISTA (temporal) ──

def crear_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_cols = ["liquidez", "cfo", "gp_ratio", "endeudamiento", "dias_efectivo"]
    for col in lag_cols:
        df[f"{col}_lag1"] = df.groupby("unidad")[col].shift(1)
    print(f"  ✅ Lag features (t-1): {len(lag_cols)} columnas creadas")
    return df


def crear_diferencias(df: pd.DataFrame) -> pd.DataFrame:
    diff_cols = ["liquidez", "gp_ratio", "endeudamiento"]
    for col in diff_cols:
        df[f"{col}_diff"] = df.groupby("unidad")[col].diff()
    print(f"  ✅ Diferencias interanuales: {len(diff_cols)} columnas creadas")
    return df


def crear_rolling(df: pd.DataFrame) -> pd.DataFrame:
    roll_cols = ["cfo", "liquidez", "dias_efectivo"]
    for col in roll_cols:
        df[f"{col}_roll2"] = df.groupby("unidad")[col].transform(
            lambda s: s.rolling(2, min_periods=1).mean()
        )
    print(f"  ✅ Rolling mean (2 años): {len(roll_cols)} columnas creadas")
    return df


# ── Features para dataset SINTÉTICO (sin temporalidad) ──

def crear_features_sintetico(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering para el dataset de 500 filas."""
    # Flag: CFO negativo
    df["cfo_negativo"] = (df["cfo"] < 0).astype(int)
    print("  ✅ Flag: cfo_negativo")

    # Flag: liquidez baja (< 1 = no cubre pasivos corrientes)
    df["liquidez_baja"] = (df["liquidez"] < 1).astype(int)
    print("  ✅ Flag: liquidez_baja")

    # Flag: días de efectivo crítico (< 30)
    df["dias_critico"] = (df["dias_efectivo"] < 30).astype(int)
    print("  ✅ Flag: dias_critico")

    # Interacciones
    df["liquidez_x_dias"] = df["liquidez"] * df["dias_efectivo"]
    df["cfo_x_hhi"] = df["cfo"] * df["hhi_fuentes"]
    df["gasto_x_hhi"] = df["gastos_personal"] * df["hhi_fuentes"]
    print("  ✅ Interacciones: liquidez×dias, cfo×hhi, gasto×hhi")

    # Ratio: CFO normalizado por gasto de personal
    df["cfo_sobre_gasto"] = df["cfo"] / df["gastos_personal"].replace(0, np.nan)
    df["cfo_sobre_gasto"] = df["cfo_sobre_gasto"].fillna(0)
    print("  ✅ Ratio: cfo / gastos_personal")

    # Concentración: HHI al cuadrado (amplifica concentración extrema)
    df["hhi_sq"] = df["hhi_fuentes"] ** 2
    print("  ✅ HHI cuadrado")

    return df


# ── Features comunes para dataset REALISTA ──

def crear_flags_interacciones_realista(df: pd.DataFrame) -> pd.DataFrame:
    df["cfo_negativo"] = (df["cfo"] < 0).astype(int)
    fuente_cols = [
        "participacion_ley30", "participacion_regalias",
        "participacion_servicios", "participacion_matriculas",
    ]
    df["top2_concentracion"] = df[fuente_cols].apply(
        lambda row: row.nlargest(2).sum(), axis=1
    )
    df["gp_x_endeudam"] = df["gp_ratio"] * df["endeudamiento"]
    df["liq_x_dias"] = df["liquidez"] * df["dias_efectivo"]
    print("  ✅ Flags: cfo_negativo, top2_concentracion")
    print("  ✅ Interacciones: gp_x_endeudam, liq_x_dias")
    return df


# ── Pipeline principal ──

def transformar(df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    """Ejecuta el pipeline de transformación adaptado al tipo de dataset."""
    es_realista = "unidad" in df.columns

    print("=" * 60)
    tipo = "REALISTA" if es_realista else "SINTÉTICO"
    print(f"  Fase 3 — Transformación y Feature Engineering ({tipo})")
    print("=" * 60)

    cols_antes = len(df.columns)

    if es_realista:
        df = crear_lag_features(df)
        df = crear_diferencias(df)
        df = crear_rolling(df)
        df = crear_flags_interacciones_realista(df)

        # Eliminar filas sin lag
        filas_antes = len(df)
        df = df.dropna().copy()
        print(f"\n  Filas eliminadas por lags: {filas_antes - len(df)}")
    else:
        df = crear_features_sintetico(df)

    print(f"  Features creados: {len(df.columns) - cols_antes}")
    print(f"  Dataset final: {df.shape[0]} filas × {df.shape[1]} columnas")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\n  💾 Guardado: {save_path}")

    print()
    return df