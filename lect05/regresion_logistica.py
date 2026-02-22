import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import reciprocal
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Crear carpeta de salida si no existe
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Cargar y transformar
df = sns.load_dataset('titanic')
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['age'] = df['age'].fillna(df['age'].median())

# 2. Determinar columna objetivo (Clasificación: Survived)
X = df[['pclass', 'sex', 'age', 'fare']]
y = df['survived']

# 3. Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Definir Pipeline
pipeline_log = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 5. Distribución de parámetros
param_dist = {
    'classifier__C': reciprocal(1e-4, 1e2),
    'classifier__solver': ['lbfgs', 'liblinear']
}

# 6. Búsqueda y Entrenamiento
log_search = RandomizedSearchCV(pipeline_log, param_dist, n_iter=50, cv=5, random_state=42)
log_search.fit(X_train, y_train)

# 7. Resultados y Guardar en TXT
y_pred = log_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

with open(f'{output_dir}/logistica_metricas.txt', 'w') as f:
    f.write("--- RESULTADOS REGRESION LOGISTICA ---\n")
    f.write(f"Mejores Parámetros: {log_search.best_params_}\n")
    f.write(f"Accuracy de prueba: {acc:.2f}\n")
    f.write(f"F1-Score de prueba: {f1:.2f}\n")

# 8. Matriz de Confusión y Guardar
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusión - Regresión Logística')
plt.savefig(f'{output_dir}/logistica_matriz_confusion.png')
plt.close()

print(f"Proceso logístico terminado. Resultados guardados en la carpeta '{output_dir}'.")