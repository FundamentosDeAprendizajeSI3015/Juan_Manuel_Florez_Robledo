import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import reciprocal
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score

# Crear carpeta de salida si no existe
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Cargar y Limpiar Datos
def cargar_datos():
    df = sns.load_dataset('titanic')
    df = df.dropna(subset=['age'])
    return df

df = cargar_datos()

# 2. Determinar columna objetivo (Regresión: Age)
X = df[['fare', 'pclass', 'sibsp']] 
y = df['age']

# 3. Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Graficar conjuntos y guardar
plt.figure(figsize=(10,6))
plt.scatter(X_train['fare'], y_train, color='blue', alpha=0.5, label='Train')
plt.scatter(X_test['fare'], y_test, color='red', alpha=0.5, label='Test')
plt.title('Regresión Lineal: Distribución de Entrenamiento y Prueba')
plt.xlabel('Fare')
plt.ylabel('Age')
plt.legend()
plt.savefig(f'{output_dir}/lineal_distribucion.png')
plt.close()

# 5. Definir Pipelines y Distribuciones
param_dist = {
    'poly__degree': [1, 2, 3],
    'regressor__alpha': reciprocal(1e-3, 1e2)
}

ridge_pipe = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(max_iter=10000)) # Aumentar a 10,000 o más
])

lasso_pipe = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Lasso())
])

# 6. Búsqueda Aleatoria y Entrenamiento
ridge_search = RandomizedSearchCV(ridge_pipe, param_dist, n_iter=50, cv=5, random_state=42)
lasso_search = RandomizedSearchCV(lasso_pipe, param_dist, n_iter=50, cv=5, random_state=42)

ridge_search.fit(X_train, y_train)
lasso_search.fit(X_train, y_train)

# 7. Resultados y Guardar en TXT
y_pred_ridge = ridge_search.predict(X_test)
y_pred_lasso = lasso_search.predict(X_test)

with open(f'{output_dir}/lineal_metricas.txt', 'w') as f:
    f.write("--- RESULTADOS REGRESION LINEAL ---\n")
    f.write(f"Mejores Parámetros Ridge: {ridge_search.best_params_}\n")
    f.write(f"Ridge MAE: {mean_absolute_error(y_test, y_pred_ridge):.2f}\n")
    f.write(f"Ridge R2: {r2_score(y_test, y_pred_ridge):.2f}\n\n")
    f.write(f"Mejores Parámetros Lasso: {lasso_search.best_params_}\n")
    f.write(f"Lasso MAE: {mean_absolute_error(y_test, y_pred_lasso):.2f}\n")
    f.write(f"Lasso R2: {r2_score(y_test, y_pred_lasso):.2f}\n")

# 8. Graficar Modelo Predicho (Ridge) y Guardar
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_ridge, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Real')
plt.ylabel('Predicho')
plt.title('Ridge: Real vs Predicho')
plt.savefig(f'{output_dir}/lineal_prediccion_ridge.png')
plt.close()

print(f"Proceso lineal terminado. Resultados guardados en la carpeta '{output_dir}'.")