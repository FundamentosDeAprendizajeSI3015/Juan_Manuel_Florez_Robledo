# =============================================================
# LAB FINTECH — VISUALIZACIÓN Y ANÁLISIS ESTADÍSTICO (EDA)
# Genera gráficos clave para entender los datos financieros.
# =============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings

# Configuración visual
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ---------------------------
# Constantes (Deben coincidir con tu archivo)
# ---------------------------
DATA_CSV = 'fintech_top_sintetico_2025.csv'
OUTPUT_DIR = Path('./plots_output')
DATE_COL = 'Month'
SPLIT_DATE = '2025-09-01' # Para visualizar el corte de Train/Test

# Columnas clave para análisis
NUM_COLS_INTEREST = [
    'Users_M', 'Revenue_USD_M', 'TPV_USD_B', 
    'TakeRate_pct', 'Marketing_Spend_USD_M', 
    'CAC_USD', 'Churn_pct', 'Close_USD'
]
CAT_COLS_INTEREST = ['Country', 'Segment', 'IsPublic']

# ---------------------------
# 1) Carga y Preparación
# ---------------------------
print("--- Cargando datos para visualización ---")
csv_path = Path(DATA_CSV)
if not csv_path.exists():
    raise FileNotFoundError(f"No se encontró {DATA_CSV}")

df = pd.read_csv(csv_path)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Crear retornos para análisis de volatilidad (igual que el script anterior)
df.sort_values(['Company', DATE_COL], inplace=True)
df['Returns'] = df.groupby('Company')['Close_USD'].pct_change().fillna(0)

# ---------------------------
# 2) Gráficas Generadas
# ---------------------------

# GRÁFICA 1: Matriz de Correlación (Heatmap)
# Útil para detectar multicolinealidad (variables repetidas)
print("Generando 1: Mapa de Calor de Correlaciones...")
plt.figure(figsize=(10, 8))
corr = df[NUM_COLS_INTEREST].corr()
mask = np.triu(np.ones_like(corr, dtype=bool)) # Ocultar la mitad superior repetida
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r', center=0, square=True)
plt.title('Mapa de Correlación entre Variables Financieras')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_correlation_heatmap.png')
plt.close()

# GRÁFICA 2: Distribuciones Numéricas (Histogramas)
# Útil para ver si los datos son normales (campana) o están sesgados
print("Generando 2: Distribuciones de Variables Clave...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(NUM_COLS_INTEREST):
    if i < len(axes):
        sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='teal', bins=30)
        axes[i].set_title(f'Distribución de {col}')
        axes[i].set_xlabel('')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_numerical_distributions.png')
plt.close()

# GRÁFICA 3: Series de Tiempo y Corte de Validación
# Muestra la tendencia y dónde cortamos los datos para el modelo
print("Generando 3: Series de Tiempo con Corte Train/Test...")
plt.figure(figsize=(14, 7))
# Graficamos el promedio de Revenue de todas las empresas para ver la tendencia global
sns.lineplot(data=df, x=DATE_COL, y='Revenue_USD_M', hue='Segment', ci=None, lw=2)
plt.axvline(pd.to_datetime(SPLIT_DATE), color='red', linestyle='--', label=f'Corte Train/Test ({SPLIT_DATE})')
plt.title('Tendencia de Ingresos por Segmento a lo largo del tiempo')
plt.legend()
plt.savefig(OUTPUT_DIR / '03_time_series_split.png')
plt.close()

# GRÁFICA 4: Boxplots para Detección de Outliers
# Útil para ver valores extremos que pueden dañar el modelo
print("Generando 4: Análisis de Outliers (Boxplots)...")
plt.figure(figsize=(14, 6))
# Normalizamos datos solo para visualizarlos juntos en el boxplot
df_norm = (df[NUM_COLS_INTEREST] - df[NUM_COLS_INTEREST].mean()) / df[NUM_COLS_INTEREST].std()
sns.boxplot(data=df_norm, orient='h', palette="Set2")
plt.title('Detección de Outliers (Datos Estandarizados)')
plt.xlabel('Desviaciones Estándar de la Media')
plt.savefig(OUTPUT_DIR / '04_outliers_boxplot.png')
plt.close()

# GRÁFICA 5: Análisis Categórico (Barras)
print("Generando 5: Conteo por Categorías...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(data=df, x='Region', ax=axes[0], palette='viridis')
axes[0].set_title('Empresas por Región')

sns.countplot(data=df, x='Segment', ax=axes[1], palette='magma')
axes[1].set_title('Empresas por Segmento')
axes[1].tick_params(axis='x', rotation=45)

# Distribución de Retornos (Riesgo)
sns.kdeplot(data=df, x='Returns', hue='IsPublic', fill=True, ax=axes[2])
axes[2].set_title('Distribución de Retornos (Riesgo)')
axes[2].set_xlim(-0.3, 0.3) # Limitar visualización para ver el centro

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_categorical_analysis.png')
plt.close()

# GRÁFICA 6: Pairplot (Relaciones Complejas)
print("Generando 6: Pairplot (Relaciones bivariadas)...")
sample_cols = ['Revenue_USD_M', 'Marketing_Spend_USD_M', 'Users_M', 'Churn_pct', 'Segment']

# CORRECCIÓN: Tomar 500 o el total de filas si hay menos de 500
n_samples = min(500, len(df)) 

sns.pairplot(df[sample_cols].sample(n=n_samples, random_state=42), hue='Segment', corner=True)
plt.savefig(OUTPUT_DIR / '06_pairplot_relationships.png')
plt.close()