import pandas as pd
import numpy as np


# FASE 1: ENTENDIMIENTO DE DATOS (Semana 2)

# El objetivo es convertir datos en programas.
# Definimos la Tarea (T): Clasificar fallos de máquina.

def cargar_y_definir():
    ruta = "data/ai4i2020.csv"
    try:
        # Cargamos el dataset usando Pandas 
        df = pd.read_csv(ruta)
        print("✅ Dataset cargado correctamente.")
        print(f"Dimensiones iniciales: {df.shape} (Filas, Columnas) [cite: 94]")
        return df
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo en 'data/datos/ai4i2020.csv'")
        return None

# FASE 2: PREPROCESAMIENTO Y LIMPIEZA (Semana 3)
# Fase crítica (70-80% del tiempo) para evitar GIGO.

def limpieza_profunda(df):
    # 1. Eliminación de identificadores irrelevantes 
    # UID y Product ID no aportan patrones reales al modelo 
    df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
    
    # 2. Manejo de Valores Faltantes (NaNs) [cite: 105]
    # Verificamos si hay nulos [cite: 107]
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        # Imputación inteligente: usamos la mediana para evitar sesgos por outliers 
        df.fillna(df.median(numeric_only=True), inplace=True)
    
    # 3. Eliminación de Duplicados 
    # Evita redundancias que afecten la precisión
    df.drop_duplicates(inplace=True)
    
    print("✅ Limpieza completada (Eliminación de IDs y manejo de nulos).")
    return df


# FASE 3: TRANSFORMACIÓN Y EXPLORACIÓN (Semana 4)
# Preparación de características (Features) para el algoritmo.

def transformar_y_explorar(df):
    # 1. Inspección Rápida [cite: 90]
    print("\n--- Estadísticas Descriptivas ---")
    print(df.describe()) # Media, min, max, percentiles [cite: 98]
    
    # 2. One-Hot Encoding (Semana 4) 
    # Transformamos la columna 'Type' (L, M, H) en variables numéricas 
    df = pd.get_dummies(df, columns=['Type'], prefix='Calidad')
    
    # 3. Ingeniería de Características (Feature Engineering)
    # Creamos una variable basada en el conocimiento experto de la Lectura 2 
    # Diferencia de temperatura entre el proceso y el aire 
    df['Dif_Temperatura'] = df['Process temperature [K]'] - df['Air temperature [K]']
    
    # 4. Consistencia y Validación Lógica [cite: 177]
    # Conservamos solo filas con lógica física (Torque positivo, por ejemplo) 
    df = df[df['Torque [Nm]'] > 0]
    
    # 5. Transformación de tipos 
    # Aseguramos que los indicadores de falla sean enteros 
    columnas_fallo = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    for col in columnas_fallo:
        df[col] = df[col].astype(int)

    print("✅ Transformación completada (Encoding y Feature Engineering).")
    return df

# EJECUCIÓN DEL PIPELINE


import os  # Necesario para manejar carpetas

if __name__ == "__main__":
    data = cargar_y_definir()
    
    if data is not None:
        # Paso 1: Limpiar
        data = limpieza_profunda(data)
        
        # Paso 2: Transformar y Explorar
        data_final = transformar_y_explorar(data)
        
        # --- SOLUCIÓN AL ERROR ---
        ruta_salida = "data/datos_procesados"
        if not os.path.exists(ruta_salida):
            os.makedirs(ruta_salida)
            print(f"📂 Carpeta creada: {ruta_salida}")
        
        # Guardar el resultado procesado
        data_final.to_csv(f"{ruta_salida}/ai4i2020_procesado.csv", index=False)
        print(f"\n🚀 Pipeline finalizado. Archivo guardado en '{ruta_salida}/ai4i2020_procesado.csv'")
        print(data_final.head())