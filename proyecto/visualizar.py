"""
=============================================================
 SCRIPT DE VISUALIZACIÓN - Pipeline Mantenimiento Predictivo
=============================================================
 Genera 4 tipos de gráficas:
   1. Árbol de decisión visual (estructura)
   2. Importancia de características (barras)
   3. Comparación de métricas entre modelos
   4. Curva ROC de ambos modelos
=============================================================
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, auc

from src.fase1_entendimiento import cargar_y_definir
from src.fase2_preprocesamiento import limpieza_profunda
from src.fase3_transformacion import transformar_y_explorar
from src.fase4_modelo import (
    preparar_features,
    balancear_muestra,
    construir_conjuntos,
    entrenar_modelo,
    evaluar_modelo,
)


def graficar_arbol(modelo_dt, feature_names):
    """1. Visualización de la estructura del árbol de decisión."""
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        modelo_dt,
        feature_names=feature_names,
        class_names=['No Fallo', 'Fallo'],
        filled=True,
        rounded=True,
        max_depth=4,  # Limitar profundidad para legibilidad
        fontsize=8,
        ax=ax,
    )
    ax.set_title('Estructura del Árbol de Decisión (max_depth=4)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/arbol_decision_estructura.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 1/4 — Árbol de decisión guardado en 'outputs/arbol_decision_estructura.png'")


def graficar_importancia(modelo_dt, modelo_rf, feature_names):
    """2. Importancia de características para ambos modelos."""
    imp_dt = modelo_dt.feature_importances_
    imp_rf = modelo_rf.feature_importances_

    # Ordenar por importancia del Random Forest
    indices = np.argsort(imp_rf)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Decision Tree
    axes[0].barh(range(len(indices)), imp_dt[indices], color='#3498db')
    axes[0].set_yticks(range(len(indices)))
    axes[0].set_yticklabels([feature_names[i] for i in indices])
    axes[0].set_xlabel('Importancia')
    axes[0].set_title('Decision Tree', fontsize=14, fontweight='bold')

    # Random Forest
    axes[1].barh(range(len(indices)), imp_rf[indices], color='#2ecc71')
    axes[1].set_yticks(range(len(indices)))
    axes[1].set_yticklabels([feature_names[i] for i in indices])
    axes[1].set_xlabel('Importancia')
    axes[1].set_title('Random Forest', fontsize=14, fontweight='bold')

    fig.suptitle('Importancia de Características', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/importancia_caracteristicas.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 2/4 — Importancia de características guardada en 'outputs/importancia_caracteristicas.png'")


def graficar_comparacion_metricas(resultados_dt, resultados_rf):
    """3. Comparación de métricas entre ambos modelos."""
    metricas = ['accuracy', 'precision', 'recall', 'f1']
    etiquetas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    valores_dt = [resultados_dt[m] for m in metricas]
    valores_rf = [resultados_rf[m] for m in metricas]

    x = np.arange(len(etiquetas))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    barras_dt = ax.bar(x - ancho/2, valores_dt, ancho, label='Decision Tree', color='#3498db')
    barras_rf = ax.bar(x + ancho/2, valores_rf, ancho, label='Random Forest', color='#2ecc71')

    # Agregar valores encima de las barras
    for barra in barras_dt:
        ax.text(barra.get_x() + barra.get_width()/2., barra.get_height() + 0.005,
                f'{barra.get_height():.3f}', ha='center', va='bottom', fontsize=10)
    for barra in barras_rf:
        ax.text(barra.get_x() + barra.get_width()/2., barra.get_height() + 0.005,
                f'{barra.get_height():.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Valor')
    ax.set_title('Comparación de Métricas — Decision Tree vs Random Forest', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/comparacion_metricas.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 3/4 — Comparación de métricas guardada en 'outputs/comparacion_metricas.png'")


def graficar_curva_roc(modelo_dt, modelo_rf, X_test, y_test):
    """4. Curva ROC de ambos modelos."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Decision Tree
    y_prob_dt = modelo_dt.predict_proba(X_test)[:, 1]
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
    auc_dt = auc(fpr_dt, tpr_dt)
    ax.plot(fpr_dt, tpr_dt, color='#3498db', lw=2, label=f'Decision Tree (AUC = {auc_dt:.3f})')

    # Random Forest
    y_prob_rf = modelo_rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    auc_rf = auc(fpr_rf, tpr_rf)
    ax.plot(fpr_rf, tpr_rf, color='#2ecc71', lw=2, label=f'Random Forest (AUC = {auc_rf:.3f})')

    # Línea diagonal (clasificador aleatorio)
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Aleatorio (AUC = 0.500)')

    ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    ax.set_title('Curva ROC — Decision Tree vs Random Forest', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/curva_roc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 4/4 — Curva ROC guardada en 'outputs/curva_roc.png'")


def main():
    print("=" * 60)
    print(" GENERANDO VISUALIZACIONES")
    print("=" * 60)

    os.makedirs("outputs", exist_ok=True)

    # Ejecutar pipeline hasta obtener modelos entrenados
    df = cargar_y_definir()
    if df is None:
        return

    df = limpieza_profunda(df)
    df = transformar_y_explorar(df)

    X, y = preparar_features(df)
    X_bal, y_bal = balancear_muestra(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = construir_conjuntos(X_bal, y_bal)

    # Entrenar modelos
    modelo_dt, nombre_dt = entrenar_modelo(X_train, y_train, tipo='decision_tree')
    modelo_rf, nombre_rf = entrenar_modelo(X_train, y_train, tipo='random_forest')

    # Evaluar para obtener métricas
    print("\n" + "=" * 60)
    print(" GENERANDO GRÁFICAS...")
    print("=" * 60 + "\n")

    resultados_dt = evaluar_modelo(modelo_dt, nombre_dt, X_test, y_test, conjunto="Test")
    resultados_rf = evaluar_modelo(modelo_rf, nombre_rf, X_test, y_test, conjunto="Test")

    feature_names = list(X_train.columns)

    # Generar las 4 gráficas
    graficar_arbol(modelo_dt, feature_names)
    graficar_importancia(modelo_dt, modelo_rf, feature_names)
    graficar_comparacion_metricas(resultados_dt, resultados_rf)
    graficar_curva_roc(modelo_dt, modelo_rf, X_test, y_test)

    print("\n🚀 Todas las visualizaciones generadas en 'outputs/':")
    print("   📊 arbol_decision_estructura.png")
    print("   📊 importancia_caracteristicas.png")
    print("   📊 comparacion_metricas.png")
    print("   📊 curva_roc.png")


if __name__ == "__main__":
    main()