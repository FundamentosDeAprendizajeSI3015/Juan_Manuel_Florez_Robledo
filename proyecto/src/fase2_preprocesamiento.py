"""
FASE 2: PREPROCESAMIENTO Y LIMPIEZA (Semana 3)
-------------------------------------------------
Fase crítica (70-80% del tiempo). Principio GIGO.
- Eliminación de identificadores irrelevantes (UDI, Product ID).
- Imputación de nulos con la mediana.
- Eliminación de duplicados.
"""
import pandas as pd


def limpieza_profunda(df):
    """Limpia el DataFrame: elimina IDs, imputa nulos, quita duplicados."""
    df = df.copy()

    # 1. Eliminación de identificadores irrelevantes
    df.drop(['UDI', 'Product ID'], axis=1, inplace=True)

    # 2. Manejo de Valores Faltantes (NaNs)
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        print(f"   ⚠️ Se encontraron {nulos} valores nulos. Imputando con mediana...")
        df.fillna(df.median(numeric_only=True), inplace=True)
    else:
        print("   ✔ Sin valores nulos.")

    # 3. Eliminación de Duplicados
    duplicados = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    if duplicados > 0:
        print(f"   ⚠️ Se eliminaron {duplicados} filas duplicadas.")

    print(f"✅ Limpieza completada. Dimensiones: {df.shape}")
    return df