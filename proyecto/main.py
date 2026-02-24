"""
=============================================================
 PIPELINE DE MANTENIMIENTO PREDICTIVO INTELIGENTE (AI4I 2020)
=============================================================
 Ejecuta las 6 fases del ciclo de vida del sistema de ML:
   Fase 1: Entendimiento y Definición
   Fase 2: Preprocesamiento y Limpieza
   Fase 3: Transformación y Exploración
   Fase 4: Espacio de Características + Balanceo + Split 60/20/20
   Fase 5: Entrenamiento del Modelo
   Fase 6: Evaluación (Validación + Test)
=============================================================
"""
import os
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


def main():
    print("=" * 60)
    print(" PIPELINE DE MANTENIMIENTO PREDICTIVO - AI4I 2020")
    print("=" * 60)

    # Crear carpetas de salida si no existen
    os.makedirs("data/datos_procesados", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # ── FASE 1: Carga ──
    print("\n📌 FASE 1: Entendimiento y Definición")
    print("-" * 40)
    df = cargar_y_definir()
    if df is None:
        return

    # ── FASE 2: Limpieza ──
    print("\n📌 FASE 2: Preprocesamiento y Limpieza")
    print("-" * 40)
    df = limpieza_profunda(df)

    # ── FASE 3: Transformación ──
    print("\n📌 FASE 3: Transformación y Exploración")
    print("-" * 40)
    df = transformar_y_explorar(df)

    # Guardar datos procesados
    df.to_csv("data/datos_procesados/ai4i2020_procesado.csv", index=False)
    print("💾 Datos procesados guardados en 'data/datos_procesados/'")

    # ── FASE 4: Features + Balanceo + Split ──
    print("\n📌 FASE 4: Espacio de Características + Balanceo")
    print("-" * 40)
    X, y = preparar_features(df)

    # Balancear: tomar 339 fallos + 339 no fallos = 678 muestras
    X_bal, y_bal = balancear_muestra(X, y)

    # Dividir 60/20/20 sobre la muestra balanceada
    X_train, X_val, X_test, y_train, y_val, y_test = construir_conjuntos(X_bal, y_bal)

    # ── FASE 5 & 6: Entrenar y Evaluar ──
    resultados = {}

    for tipo in ['decision_tree', 'random_forest']:
        nombre_corto = tipo.replace('_', ' ').title().replace(' ', '')
        print(f"\n📌 FASE 5-6: {tipo.replace('_', ' ').title()}")
        print("-" * 40)

        modelo, nombre = entrenar_modelo(X_train, y_train, tipo=tipo)

        # Evaluar en Validación
        print(f"\n  ── Evaluación en VALIDACIÓN ──")
        evaluar_modelo(
            modelo, nombre, X_val, y_val,
            conjunto="Validación",
            guardar_fig=f"outputs/confusion_{tipo}_val.png"
        )

        # Evaluar en Test
        print(f"\n  ── Evaluación en TEST ──")
        resultados[nombre] = evaluar_modelo(
            modelo, nombre, X_test, y_test,
            conjunto="Test",
            guardar_fig=f"outputs/confusion_{tipo}_test.png"
        )

    # ── RESUMEN FINAL ──
    print("\n" + "=" * 60)
    print(" RESUMEN DE RESULTADOS (TEST)")
    print("=" * 60)
    for nombre, metricas in resultados.items():
        print(f"\n  {nombre}:")
        for metrica, valor in metricas.items():
            print(f"    {metrica:>10}: {valor:.4f}")

    print("\n🚀 Pipeline finalizado exitosamente.")
    print("   📂 Datos procesados: data/datos_procesados/")
    print("   📊 Gráficas:         outputs/")


if __name__ == "__main__":
    main()