"""
FASE 3: TRANSFORMACIÓN Y EXPLORACIÓN (Semana 4)
--------------------------------------------------
- One-Hot Encoding de la variable 'Type' (L, M, H).
- Feature Engineering: Diferencia de temperatura.
- Validación lógica (Torque > 0).
- Conversión de tipos para columnas de fallo.
"""
import pandas as pd


def transformar_y_explorar(df):
    """Transforma el DataFrame: encoding, feature engineering, validación."""
    df = df.copy()

    # 1. Inspección Rápida
    print("\n--- Estadísticas Descriptivas ---")
    print(df.describe())

    # 2. One-Hot Encoding para 'Type'
    df = pd.get_dummies(df, columns=['Type'], prefix='Calidad')

    # 3. Feature Engineering: Diferencia de temperatura
    df['Dif_Temperatura'] = df['Process temperature [K]'] - df['Air temperature [K]']

    # 4. Validación Lógica (Torque positivo)
    filas_antes = len(df)
    df = df[df['Torque [Nm]'] > 0]
    eliminadas = filas_antes - len(df)
    if eliminadas > 0:
        print(f"   ⚠️ Se eliminaron {eliminadas} filas con Torque <= 0.")

    # 5. Asegurar tipos enteros en columnas de fallo
    columnas_fallo = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    for col in columnas_fallo:
        df[col] = df[col].astype(int)

    # Asegurar que las columnas de One-Hot sean int (no bool)
    for col in df.columns:
        if col.startswith('Calidad_'):
            df[col] = df[col].astype(int)

    print(f"✅ Transformación completada. Dimensiones: {df.shape}")
    return df