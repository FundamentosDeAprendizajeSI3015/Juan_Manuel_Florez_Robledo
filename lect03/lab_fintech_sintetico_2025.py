# =============================================================
# LAB FINTECH (SINTÉTICO 2025) — PREPROCESAMIENTO Y EDA
# =============================================================
# NOTA DEL ESTUDIANTE:
# Este script es nuestra "fábrica" de datos. Entra un CSV sucio y salen
# archivos .parquet limpios listos para entrenar una IA.
# No hay que pasarle argumentos, solo darle 'run'.
# =============================================================

import json
from pathlib import Path
import warnings
# Ignoramos advertencias molestas para mantener la consola limpia
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Constantes de la práctica
# ---------------------------
# Definimos nombres de archivos y rutas aquí para no equivocarnos abajo.
DATA_CSV = 'fintech_top_sintetico_2025.csv'
DATA_DICT = 'fintech_top_sintetico_dictionary.json'
OUTDIR = Path('./data_output_finanzas_sintetico')

# ¡OJO AQUÍ! Esta fecha es clave. En finanzas NO podemos dividir datos
# aleatoriamente (random shuffle) porque romperíamos el tiempo.
# Todo lo que pase ANTES de esta fecha será para aprender (Train),
# y lo que pase DESPUÉS será para evaluar (Test).
SPLIT_DATE = '2025-09-01'  

# Definimos qué columnas son qué.
DATE_COL = 'Month'
ID_COLS = ['Company'] # Identificador, no sirve para predecir, solo para organizar.
CAT_COLS = ['Country', 'Region', 'Segment', 'Subsegment', 'IsPublic', 'Ticker']
# Estas son las variables numéricas que vamos a limpiar y escalar.
NUM_COLS = [
    'Users_M','NewUsers_K','TPV_USD_B','TakeRate_pct','Revenue_USD_M',
    'ARPU_USD','Churn_pct','Marketing_Spend_USD_M','CAC_USD','CAC_Total_USD_M',
    'Close_USD','Private_Valuation_USD_B'
]
PRICE_COLS = ['Close_USD']  # Usaremos esto para calcular rendimientos (ganancias/pérdidas).

# ---------------------------
# 0) Carga de diccionario
# ---------------------------
print("\n=== 0) Cargando diccionario de datos ===")
dict_path = Path(DATA_DICT)
# Verificación de seguridad: si no está el archivo, paramos todo.
if not dict_path.exists():
    raise FileNotFoundError(f"No se encontró {DATA_DICT}. Asegúrate de tener el archivo en la misma carpeta.")

with open(dict_path, 'r', encoding='utf-8') as f:
    data_dict = json.load(f)
print("Descripción:", data_dict.get('description', '(sin descripción)'))
print("Periodo:", data_dict.get('period', '(desconocido)'))

# ---------------------------
# 1) Carga del CSV
# ---------------------------
print("\n=== 1) Cargando CSV sintético ===")
csv_path = Path(DATA_CSV)
if not csv_path.exists():
    raise FileNotFoundError(f"No se encontró {DATA_CSV}. Asegúrate de tener el archivo en la misma carpeta.")

df = pd.read_csv(csv_path)
print("Shape:", df.shape) # Nos dice cuántas filas y columnas tenemos al inicio.

# Verificamos que exista la columna de fecha, si no, no podemos hacer series de tiempo.
if DATE_COL not in df.columns:
    raise KeyError(f"La columna de fecha '{DATE_COL}' no existe en el CSV.")

# TRUCO IMPORTANTE:
# Convertimos la columna 'Month' a formato fecha real (datetime) y ordenamos.
# Si los datos no están ordenados cronológicamente, calcular retornos o tendencias
# daría resultados basura. Ordenamos por Fecha y luego por Empresa.
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
df = df.sort_values([DATE_COL] + ID_COLS).reset_index(drop=True)

print("Primeras filas:")
print(df.head(3))

# ---------------------------
# 2) EDA breve (Análisis Exploratorio)
# ---------------------------
print("\n=== 2) EDA rápido ===")
# Esto es para ver qué tipo de datos tenemos y cuánta memoria usamos.
print("Info:")
print(df.info())
# Aquí vemos cuántos datos faltan (Nulos). Es vital para saber qué limpiar.
print("\nNulos por columna (top 15):")
print(df.isna().sum().sort_values(ascending=False).head(15))

# ---------------------------
# 3) Limpieza básica
# ---------------------------
print("\n=== 3) Limpieza ===")
# Bucle para limpiar columnas NUMÉRICAS:
for c in NUM_COLS:
    if c in df.columns and df[c].isna().any():
        # Nos aseguramos que sean números
        df[c] = pd.to_numeric(df[c], errors='coerce')
        # Rellenamos los huecos con la MEDIANA.
        # ¿Por qué mediana y no promedio? Porque en finanzas los "unicornios" (valores extremos)
        # pueden inflar el promedio falsamente. La mediana es más honesta aquí.
        df[c] = df[c].fillna(df[c].median())

# Bucle para limpiar columnas de TEXTO (Categóricas):
for c in CAT_COLS:
    if c in df.columns and df[c].isna().any():
        # Si falta el dato (ej: no sabemos la Región), le ponemos una etiqueta explícita.
        # Así el modelo aprende que "no tener región" quizás significa algo.
        df[c] = df[c].fillna('__MISSING__')

# ---------------------------
# 4) Ingeniería ligera: retornos
# ---------------------------
print("\n=== 4) Ingeniería de rasgos (retornos) ===")
# Aquí creamos nuevas variables matemáticas útiles para predecir.
if all([pc in df.columns for pc in PRICE_COLS]):
    for pc in PRICE_COLS:
        # Calculamos el retorno porcentual: (Precio_Hoy - Precio_Ayer) / Precio_Ayer
        # Usamos .groupby(ID_COLS) para que NO calcule el cambio de precio entre
        # la "Empresa A" y la "Empresa B", sino solo dentro de la misma empresa.
        df[pc + '_ret'] = (
            df.sort_values([ID_COLS[0], DATE_COL])
              .groupby(ID_COLS)[pc]
              .pct_change()
        )
        # Calculamos log-retornos (a los estadísticos les gustan porque se suman mejor).
        df[pc + '_logret'] = np.log1p(df[pc + '_ret'])
        
        # El primer mes de cada empresa no tiene "mes anterior", así que da NaN.
        # Los rellenamos con 0 para no perder esa fila.
        df[pc + '_ret'] = df[pc + '_ret'].fillna(0.0)
        df[pc + '_logret'] = df[pc + '_logret'].fillna(0.0)
else:
    print("[INFO] Columnas de precio no disponibles; se omite cálculo de retornos.")

# Agregamos estas nuevas columnas calculadas a nuestra lista de variables numéricas.
extra_num = [c for c in [pc + '_ret' for pc in PRICE_COLS] + [pc + '_logret' for pc in PRICE_COLS] if c in df.columns]
NUM_USED = [c for c in NUM_COLS if c in df.columns] + extra_num

# ---------------------------
# 5) Separación X / y y Codificación
# ---------------------------
print("\n=== 5) Preparación de X: codificación one-hot y escalado ===")
# Eliminamos la Fecha y el Nombre de la empresa.
# El modelo debe aprender patrones numéricos, no memorizar nombres.
X = df.drop(columns=[DATE_COL] + ID_COLS, errors='ignore').copy()

# One-Hot Encoding:
# Convertimos texto a números. Ejemplo: Columna "Región" con valor "Latam"
# se convierte en una columna "Region_Latam" con valor 1.
cat_in_X = [c for c in CAT_COLS if c in X.columns]
X = pd.get_dummies(X, columns=cat_in_X, drop_first=True)

# DIVISIÓN TEMPORAL (TIME SPLIT):
# Aquí hacemos el corte del tiempo.
cutoff = pd.to_datetime(SPLIT_DATE)

# Creamos máscaras (filtros) booleanos.
idx_train = df[DATE_COL] < cutoff  # Pasado (Entrenamiento)
idx_test = df[DATE_COL] >= cutoff  # Futuro (Prueba)

X_train, X_test = X.loc[idx_train].copy(), X.loc[idx_test].copy()

# ESCALADO (StandardScaler):
# Ponemos todos los números en la misma escala (media 0, desviación 1).
# OJO: Hacemos .fit() SOLO con X_train.
# ¿Por qué? Para simular la vida real. No conocemos el futuro (Test), así que
# no podemos usar la media del futuro para escalar el pasado. Eso sería trampa ("Data Leakage").
num_in_X = [c for c in NUM_USED if c in X_train.columns]
scaler = StandardScaler()

if num_in_X:
    X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X]) # Aprende y transforma Train
    X_test[num_in_X] = scaler.transform(X_test[num_in_X])       # Solo transforma Test usando lo aprendido
else:
    print("[INFO] No se encontraron columnas numéricas para escalar.")

print("Shapes -> X_train:", X_train.shape, " X_test:", X_test.shape)

# ---------------------------
# 6) Exportación
# ---------------------------
print("\n=== 6) Exportación ===")
# Creamos la carpeta de salida si no existe.
OUTDIR.mkdir(parents=True, exist_ok=True)
train_path = OUTDIR / 'fintech_train.parquet'
test_path = OUTDIR / 'fintech_test.parquet'

# Guardamos en formato PARQUET.
# Es mejor que CSV porque guarda los tipos de datos (sabe qué es número y qué es texto)
# y pesa mucho menos.
X_train.to_parquet(train_path, index=False)
X_test.to_parquet(test_path, index=False)

# Guardamos un "log" o esquema de lo que acabamos de hacer en un JSON.
# Esto sirve para auditoría: saber qué columnas usamos, dónde cortamos el tiempo, etc.
processed_schema = {
    'source_csv': str(csv_path.resolve()),
    'source_dict': str(dict_path.resolve()),
    'date_col': DATE_COL,
    'id_cols': ID_COLS,
    'categorical_cols_used': cat_in_X,
    'numeric_cols_used': num_in_X,
    'engineered_cols': extra_num,
    'split': {
        'type': 'time_split',
        'cutoff': SPLIT_DATE,
        'train_rows': int(idx_train.sum()),
        'test_rows': int(idx_test.sum()),
    },
    'X_train_shape': list(X_train.shape),
    'X_test_shape': list(X_test.shape),
    'notes': [
        'Dataset 100% SINTÉTICO con fines académicos; no refleja métricas reales.',
        'Evitar fuga de datos: el escalador se ajusta en TRAIN y se aplica a TEST.'
    ]
}

with open(OUTDIR / 'processed_schema.json', 'w', encoding='utf-8') as f:
    json.dump(processed_schema, f, ensure_ascii=False, indent=2)

# También guardamos la lista de nombres de columnas en un txt simple.
# Útil si luego queremos ver qué 'features' entraron al modelo sin abrir el parquet.
with open(OUTDIR / 'features_columns.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(X_train.columns))

print("\nArchivos exportados:")
print(" -", train_path)
print(" -", test_path)
print(" -", OUTDIR / 'processed_schema.json')
print(" -", OUTDIR / 'features_columns.txt')

print("\n✔ Listo. Recuerda: este dataset es sintético para práctica académica.")