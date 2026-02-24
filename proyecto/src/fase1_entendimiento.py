"""
FASE 1: ENTENDIMIENTO Y DEFINICIÓN (Semana 2)
-----------------------------------------------
Definición formal (T, P, E) según Tom Mitchell:
  - Tarea (T): Clasificar si una instancia indica un fallo de máquina.
  - Rendimiento (P): Precisión de la clasificación.
  - Experiencia (E): Dataset AI4I 2020 con 10,000 ejemplos etiquetados.
Categoría: Aprendizaje Supervisado.
"""
import pandas as pd


def cargar_y_definir(ruta="data/ai4i2020.csv"):
    """Carga el dataset y muestra información inicial."""
    try:
        df = pd.read_csv(ruta)
        print("✅ Dataset cargado correctamente.")
        print(f"   Dimensiones iniciales: {df.shape} (Filas, Columnas)")
        print(f"   Columnas: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en '{ruta}'")
        return None