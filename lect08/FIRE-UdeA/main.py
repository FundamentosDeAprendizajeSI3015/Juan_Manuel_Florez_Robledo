"""
FIRE-UdeA — Pipeline Completo (Ambos Datasets)
================================================
Ejecuta las 6 fases del ciclo de vida ML para:
  1. Dataset Sintético (500 filas, sin temporalidad)
  2. Dataset Realista (80 filas, panel temporal)

Uso:
    python main.py
"""
import sys
sys.path.insert(0, "src")

from fase1_entendimiento import definir_problema, cargar_datos
from fase2_preprocesamiento import preprocesar
from fase3_transformacion import transformar
from fase4_features import separar_features_target
from fase5_entrenamiento import entrenar_y_comparar
from fase6_evaluacion import evaluar_test, generar_graficas, scores_por_unidad


def ejecutar_pipeline(filepath: str, nombre: str, output_dir: str):
    """Ejecuta el pipeline completo para un dataset."""
    print("\n" + "═" * 60)
    print(f"  PIPELINE: {nombre}")
    print("═" * 60 + "\n")

    # Fase 1
    df = cargar_datos(filepath)

    # Fase 2
    es_realista = "unidad" in df.columns
    prefijo = "realista" if es_realista else "sintetico"
    df = preprocesar(df, save_path=f"data/processed/{prefijo}_limpio.csv")

    # Fase 3
    df = transformar(df, save_path=f"data/processed/{prefijo}_features.csv")

    # Fase 4
    data = separar_features_target(df)

    # Fase 5
    training = entrenar_y_comparar(data)

    # Fase 6
    evaluacion = evaluar_test(data, training)
    generar_graficas(data, training, evaluacion, output_dir=output_dir)
    scores = scores_por_unidad(data, evaluacion)

    scores.to_csv(f"{output_dir}/{prefijo}_scores_test.csv", index=False)
    print(f"  💾 Guardado: {output_dir}/{prefijo}_scores_test.csv")

    return evaluacion


def main():
    print("\n" + "█" * 60)
    print("  FIRE-UdeA — Pipeline de Riesgo Financiero")
    print("  Ejecutando para AMBOS datasets")
    print("█" * 60)

    # Fase 1: Definición del problema (una sola vez)
    definir_problema()

    # Pipeline 1: Dataset Sintético (500 filas)
    eval_sint = ejecutar_pipeline(
        filepath="data/raw/dataset_sintetico_FIRE_UdeA.csv",
        nombre="DATASET SINTÉTICO (500 filas)",
        output_dir="outputs",
    )

    # Pipeline 2: Dataset Realista (80 filas, panel temporal)
    eval_real = ejecutar_pipeline(
        filepath="data/raw/dataset_sintetico_FIRE_UdeA_realista.csv",
        nombre="DATASET REALISTA (80 filas, panel temporal)",
        output_dir="outputs",
    )

    # Resumen final comparativo
    print("\n" + "█" * 60)
    print("  RESUMEN FINAL — Ambos Datasets")
    print("█" * 60)
    print(f"\n  {'Métrica':<15} {'Sintético (500)':>15} {'Realista (80)':>15}")
    print(f"  {'-'*47}")
    print(f"  {'ROC-AUC':<15} {eval_sint['auc']:>15.4f} {eval_real['auc']:>15.4f}")
    print(f"  {'F1-Score':<15} {eval_sint['f1']:>15.4f} {eval_real['f1']:>15.4f}")
    print(f"  {'Brier':<15} {eval_sint['brier']:>15.4f} {eval_real['brier']:>15.4f}")
    print(f"  {'Accuracy':<15} {eval_sint['accuracy']:>15.1%} {eval_real['accuracy']:>15.1%}")

    print("\n" + "█" * 60)
    print("  ✅ Pipeline completado exitosamente para ambos datasets.")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()