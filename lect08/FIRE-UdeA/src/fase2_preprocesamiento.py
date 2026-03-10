"""
FIRE-UdeA — Fase 2: Preprocesamiento y Limpieza
=================================================
Principio GIGO (Garbage In, Garbage Out):
- Detección y reporte de valores nulos
- Imputación temporal (panel) o por mediana (general)
- Detección de duplicados
- Validación de integridad post-limpieza
"""
import os
import pandas as pd
import numpy as np


def reportar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Reporta valores nulos por columna."""
    nulls = df.isnull().sum()
    nulls_pos = nulls[nulls > 0]

    print("=" * 60)
    print("  Fase 2a — Detección de Valores Nulos")
    print("=" * 60)

    if len(nulls_pos) == 0:
        print("  ✅ No se encontraron valores nulos.")
    else:
        total_celdas = df.size
        total_nulos = nulls_pos.sum()
        print(f"  Total nulos: {total_nulos} de {total_celdas} celdas ({total_nulos/total_celdas*100:.1f}%)\n")
        for col, count in nulls_pos.items():
            pct = count / len(df) * 100
            print(f"    {col:30s} → {count} nulos ({pct:.1f}%)")

    # Duplicados
    dupes = df.duplicated().sum()
    print(f"\n  Filas duplicadas: {dupes}")
    if dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  → Eliminadas. Filas restantes: {len(df)}")

    print()
    return nulls_pos


def imputar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Imputación adaptada al tipo de dataset."""
    es_realista = "unidad" in df.columns

    if df.isnull().sum().sum() == 0:
        print("  ✅ Sin nulos — no se requiere imputación.\n")
        return df

    print("=" * 60)
    print("  Fase 2b — Imputación de Valores Nulos")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("label")

    if es_realista:
        df = df.sort_values(["unidad", "anio"]).reset_index(drop=True)
        for col in numeric_cols:
            antes = df[col].isnull().sum()
            df[col] = df.groupby("unidad")[col].transform(lambda s: s.ffill().bfill())
            despues = df[col].isnull().sum()
            if antes > 0:
                print(f"  {col}: {antes} → {despues} nulos (ffill+bfill por unidad)")
    else:
        for col in numeric_cols:
            antes = df[col].isnull().sum()
            if antes > 0:
                mediana = df[col].median()
                df[col] = df[col].fillna(mediana)
                print(f"  {col}: {antes} nulos imputados con mediana ({mediana:.4f})")

    # Residuales
    residuales = df[numeric_cols].isnull().sum().sum()
    if residuales > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        print(f"\n  Mediana global aplicada a {residuales} nulos residuales.")

    print(f"\n  ✅ Nulos restantes: {df.isnull().sum().sum()}")
    print()

    return df


def preprocesar(df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    """Ejecuta el pipeline completo de preprocesamiento."""
    reportar_nulos(df)
    df = imputar_nulos(df)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"  💾 Guardado: {save_path}")

    return df