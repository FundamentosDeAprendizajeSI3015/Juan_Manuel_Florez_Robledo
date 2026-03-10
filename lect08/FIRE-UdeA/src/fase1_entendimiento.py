"""
FIRE-UdeA — Fase 1: Entendimiento y Definición del Problema
============================================================
Soporta dos datasets:
- Sintético (500 filas): sin estructura temporal, 7 features
- Realista (80 filas): panel temporal, 8 unidades × 10 años
"""
import pandas as pd


def definir_problema():
    """Definición formal del problema según Tom Mitchell (T, P, E)."""
    definicion = {
        "Tarea (T)": "Clasificar si una unidad académica/administrativa de la UdeA "
                      "presentará tensión financiera (cash stress) en el periodo t+1.",
        "Rendimiento (P)": "ROC-AUC, F1-Score, Brier Score sobre conjunto de test.",
        "Experiencia (E)": "Dos datasets sintéticos con indicadores financieros etiquetados: "
                           "uno de 500 registros (general) y uno de 80 registros (panel temporal).",
        "Categoría": "Aprendizaje Supervisado — Clasificación Binaria.",
    }

    print("=" * 60)
    print("  FIRE-UdeA — Definición Formal del Problema")
    print("=" * 60)
    for key, value in definicion.items():
        print(f"\n  {key}:")
        print(f"    {value}")
    print()

    return definicion


def cargar_datos(filepath: str) -> pd.DataFrame:
    """Carga un dataset y muestra resumen inicial."""
    df = pd.read_csv(filepath)
    es_realista = "unidad" in df.columns
    nombre = "REALISTA (panel temporal)" if es_realista else "SINTÉTICO (500 registros)"

    print("=" * 60)
    print(f"  Carga del Dataset — {nombre}")
    print("=" * 60)
    print(f"  Archivo:     {filepath}")
    print(f"  Shape:       {df.shape[0]} filas × {df.shape[1]} columnas")

    if es_realista:
        print(f"  Unidades:    {df['unidad'].nunique()}")
        print(f"  Rango años:  {df['anio'].min()} - {df['anio'].max()}")

    print(f"  Prevalencia:  label=1 → {df['label'].mean():.3f} ({df['label'].sum()}/{len(df)})")
    print(f"\n  Columnas: {list(df.columns)}")
    print()

    return df