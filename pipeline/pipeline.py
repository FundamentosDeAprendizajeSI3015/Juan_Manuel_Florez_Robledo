import pandas as pd
import numpy as np

# ==========================================
# FASE 1: ENTENDIMIENTO DE DATOS (Semana 2)
# ==========================================
# El objetivo es convertir datos en programas[cite: 343].
# Definimos la Tarea (T): Clasificar fallos de máquina[cite: 371].

def cargar_y_definir():
    ruta = "data/datos/ai4i2020.csv"
    try:
        # Cargamos el dataset usando Pandas [cite: 71, 75]
        df = pd.read_csv(ruta)
        print("✅ Dataset cargado correctamente.")
        print(f"Dimensiones iniciales: {df.shape} (Filas, Columnas) [cite: 94]")
        return df
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo en 'data/datos/ai4i2020.csv'")
        return None

# ==========================================
# FASE 2: PREPROCESAMIENTO Y LIMPIEZA (Semana 3)
# ==========================================
# Fase crítica (70-80% del tiempo) para evitar GIGO[cite: 10, 14].

def limpieza_profunda(df):
    # 1. Eliminación de identificadores irrelevantes [cite: 146, 147]
    # UID y Product ID no aportan patrones reales al modelo 
    df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
    
    # 2. Manejo de Valores Faltantes (NaNs) [cite: 105]
    # Verificamos si hay nulos [cite: 107]
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        # Imputación inteligente: usamos la mediana para evitar sesgos por outliers 
        df.fillna(df.median(numeric_only=True), inplace=True)
    
    # 3. Eliminación de Duplicados [cite: 171, 172]
    # Evita redundancias que afecten la precisión [cite: 17]
    df.drop_duplicates(inplace=True)
    
    print("✅ Limpieza completada (Eliminación de IDs y manejo de nulos).")
    return df

# ==========================================
# FASE 3: TRANSFORMACIÓN Y EXPLORACIÓN (Semana 4)
# ==========================================
# Preparación de características (Features) para el algoritmo[cite: 188].

def transformar_y_explorar(df):
    # 1. Inspección Rápida [cite: 90]
    print("\n--- Estadísticas Descriptivas ---")
    print(df.describe()) # Media, min, max, percentiles [cite: 98]
    
    # 2. One-Hot Encoding (Semana 4) [cite: 243]
    # Transformamos la columna 'Type' (L, M, H) en variables numéricas 
    df = pd.get_dummies(df, columns=['Type'], prefix='Calidad')
    
    # 3. Ingeniería de Características (Feature Engineering)
    # Creamos una variable basada en el conocimiento experto de la Lectura 2 [cite: 323]
    # Diferencia de temperatura entre el proceso y el aire 
    df['Dif_Temperatura'] = df['Process temperature [K]'] - df['Air temperature [K]']
    
    # 4. Consistencia y Validación Lógica [cite: 177]
    # Conservamos solo filas con lógica física (Torque positivo, por ejemplo) [cite: 178, 179]
    df = df[df['Torque [Nm]'] > 0]
    
    # 5. Transformación de tipos [cite: 189]
    # Aseguramos que los indicadores de falla sean enteros [cite: 191]
    columnas_fallo = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    for col in columnas_fallo:
        df[col] = df[col].astype(int)

    print("✅ Transformación completada (Encoding y Feature Engineering).")
    return df

# ==========================================
# EJECUCIÓN DEL PIPELINE
# ==========================================

if __name__ == "__main__":
    data = cargar_y_definir()
    
    if data is not None:
        # Paso 1: Limpiar
        data = limpieza_profunda(data)
        
        # Paso 2: Transformar y Explorar
        data_final = transformar_y_explorar(data)
        
        # Guardar el resultado procesado
        data_final.to_csv("data/datos/ai4i2020_procesado.csv", index=False)
        print("\n🚀 Pipeline finalizado. Archivo listo en 'data/datos/ai4i2020_procesado.csv'")
        print(data_final.head()) # Ver primeras 5 filas del resultado final [cite: 92]