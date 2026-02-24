"""
FASE 4: ESPACIO DE CARACTERÍSTICAS (Semana 5)
-----------------------------------------------
- Separar X (features) e y (target).
- Balanceo: submuestreo de la clase mayoritaria (misma cantidad que fallos).
- Construir conjuntos 60% train / 20% validación / 20% test.

FASE 5: ENTRENAMIENTO DEL MODELO (Semana 6)
-----------------------------------------------
- Entrenar un modelo base (Decision Tree).
- Entrenar un modelo mejorado (Random Forest).

FASE 6: EVALUACIÓN (Semana 7)
-----------------------------------------------
- Métricas: Accuracy, Precision, Recall, F1-Score.
- Matriz de confusión.
- Reporte de clasificación.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# FASE 4: PREPARAR ESPACIO DE CARACTERÍSTICAS
# ============================================================
def preparar_features(df, target='Machine failure'):
    """
    Separa el DataFrame en X (features) e y (target).
    Excluye las columnas de sub-tipos de fallo para evitar data leakage.
    """
    columnas_excluir = [target, 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    columnas_excluir = [c for c in columnas_excluir if c in df.columns]

    X = df.drop(columns=columnas_excluir)
    y = df[target]

    print(f"\n✅ Espacio de características preparado.")
    print(f"   Features (X): {X.shape[1]} columnas -> {list(X.columns)}")
    print(f"   Target  (y): '{target}' | Distribución ORIGINAL:")
    print(f"      0 (No fallo): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"      1 (Fallo):    {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

    return X, y


def balancear_muestra(X, y, random_state=42):
    """
    Submuestreo de la clase mayoritaria para igualar la clase minoritaria.
    Si hay 339 fallos, se toman 339 no-fallos al azar -> 678 muestras totales.
    """
    df_temp = X.copy()
    df_temp['_target'] = y.values

    df_fallo = df_temp[df_temp['_target'] == 1]
    df_no_fallo = df_temp[df_temp['_target'] == 0]

    n_fallos = len(df_fallo)

    # Submuestrear la clase mayoritaria
    df_no_fallo_sub = df_no_fallo.sample(n=n_fallos, random_state=random_state)

    # Unir y mezclar
    df_balanceado = pd.concat([df_fallo, df_no_fallo_sub]).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    X_bal = df_balanceado.drop(columns=['_target'])
    y_bal = df_balanceado['_target']

    print(f"\n✅ Muestra balanceada (submuestreo).")
    print(f"   Total: {len(y_bal)} muestras ({n_fallos} fallos + {n_fallos} no fallos)")
    print(f"      0 (No fallo): {(y_bal == 0).sum()} ({(y_bal == 0).mean()*100:.1f}%)")
    print(f"      1 (Fallo):    {(y_bal == 1).sum()} ({(y_bal == 1).mean()*100:.1f}%)")

    return X_bal, y_bal


def construir_conjuntos(X, y, random_state=42):
    """
    Divide X e y en 60% train / 20% validación / 20% test.
    Usa stratify para mantener la proporción 50/50 en cada conjunto.
    """
    # Primer split: 60% train, 40% temporal
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.70,
        random_state=random_state,
        stratify=y
    )

    # Segundo split: del 40% temporal -> 50/50 = 20% val + 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    print(f"\n✅ Conjuntos construidos (60/20/20).")
    print(f"   Entrenamiento: {X_train.shape[0]} muestras (fallos: {(y_train == 1).sum()})")
    print(f"   Validación:    {X_val.shape[0]} muestras (fallos: {(y_val == 1).sum()})")
    print(f"   Prueba:        {X_test.shape[0]} muestras (fallos: {(y_test == 1).sum()})")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# FASE 5: ENTRENAMIENTO DEL MODELO
# ============================================================
def entrenar_modelo(X_train, y_train, tipo='random_forest'):
    """
    Entrena un modelo de clasificación.
    tipo: 'decision_tree' o 'random_forest'
    """
    if tipo == 'decision_tree':
        modelo = DecisionTreeClassifier(random_state=42, max_depth=10)
        nombre = "Decision Tree"
    else:
        modelo = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=15,
            n_jobs=-1
        )
        nombre = "Random Forest"

    modelo.fit(X_train, y_train)
    print(f"\n✅ Modelo '{nombre}' entrenado correctamente.")

    return modelo, nombre


# ============================================================
# FASE 6: EVALUACIÓN
# ============================================================
def evaluar_modelo(modelo, nombre, X_eval, y_eval, conjunto="Test", guardar_fig=None):
    """
    Evalúa el modelo con métricas y genera la matriz de confusión.
    conjunto: nombre del conjunto para el título (ej: "Validación", "Test").
    """
    y_pred = modelo.predict(X_eval)

    acc = accuracy_score(y_eval, y_pred)
    prec = precision_score(y_eval, y_pred, zero_division=0)
    rec = recall_score(y_eval, y_pred, zero_division=0)
    f1 = f1_score(y_eval, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f" EVALUACIÓN: {nombre} [{conjunto}]")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n--- Reporte de Clasificación ---")
    print(classification_report(y_eval, y_pred, target_names=['No Fallo', 'Fallo']))

    if guardar_fig:
        cm = confusion_matrix(y_eval, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Fallo', 'Fallo'],
            yticklabels=['No Fallo', 'Fallo'],
            ax=ax
        )
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        ax.set_title(f'Matriz de Confusión - {nombre} [{conjunto}]')
        plt.tight_layout()
        plt.savefig(guardar_fig, dpi=150)
        plt.close()
        print(f"📊 Matriz guardada en '{guardar_fig}'")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}