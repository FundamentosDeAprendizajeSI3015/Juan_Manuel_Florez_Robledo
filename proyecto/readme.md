# Pipeline de Mantenimiento Predictivo Inteligente (AI4I 2020)

Proyecto académico que implementa las fases del ciclo de vida de un sistema de Aprendizaje Automático, aplicado a la predicción de fallos en una planta de fresado industrial.

## Dataset

**AI4I 2020 Predictive Maintenance** — Kaggle

- 10,000 registros con 14 características técnicas
- Variables: Temperaturas (aire y proceso en K), velocidad de rotación (rpm), torque (Nm), desgaste de herramienta (min)
- Etiqueta objetivo: `Machine failure` (clasificación binaria)
- Tipos de fallo: TWF, HDF, PWF, OSF, RNF

## Estructura del Proyecto

```
proyecto/
├── data/
│   ├── ai4i2020.csv                  # Dataset original
│   └── datos_procesados/             # CSV limpio y transformado
├── outputs/                          # Matrices de confusión (.png)
├── src/
│   ├── __init__.py
│   ├── fase1_entendimiento.py        # Carga y definición del problema
│   ├── fase2_preprocesamiento.py     # Limpieza de datos
│   ├── fase3_transformacion.py       # Encoding y feature engineering
│   └── fase4_modelo.py              # Features, balanceo, entrenamiento y evaluación
├── main.py                           # Ejecuta todo el pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## Fases Implementadas

### Fase 1 — Entendimiento y Definición

Definición formal según Tom Mitchell (T, P, E):

- **Tarea (T):** Clasificar si una instancia indica un fallo de máquina
- **Rendimiento (P):** Precisión de la clasificación
- **Experiencia (E):** Dataset de 10,000 ejemplos etiquetados
- **Categoría:** Aprendizaje Supervisado

### Fase 2 — Preprocesamiento y Limpieza

Principio GIGO (Garbage In, Garbage Out):

- Eliminación de identificadores irrelevantes (`UDI`, `Product ID`)
- Detección e imputación de valores nulos con la mediana
- Eliminación de filas duplicadas

### Fase 3 — Transformación y Exploración

- **One-Hot Encoding:** Variable `Type` (L, M, H) convertida a columnas binarias (`Calidad_L`, `Calidad_M`, `Calidad_H`)
- **Feature Engineering:** Creación de `Dif_Temperatura` (Process - Air) como indicador de fallas térmicas (HDF)
- **Validación lógica:** Eliminación de registros con Torque ≤ 0
- **Conversión de tipos:** Columnas de fallo como enteros

### Fase 4 — Espacio de Características y Balanceo

- Separación de Features (X) y Target (y)
- Exclusión de columnas de sub-fallo (TWF, HDF, PWF, OSF, RNF) para evitar data leakage
- **Balanceo por submuestreo:** 339 fallos + 339 no fallos = 678 muestras (50/50)
- **División 60/20/20:** Entrenamiento (406), Validación (136), Prueba (136) con stratify

### Fase 5 — Entrenamiento

Dos modelos entrenados:

- **Decision Tree** (max_depth=10)
- **Random Forest** (100 estimadores, max_depth=15)

### Fase 6 — Evaluación

Métricas calculadas: Accuracy, Precision, Recall, F1-Score, Reporte de Clasificación y Matriz de Confusión.

## Resultados (Conjunto de Test)

| Modelo | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree | 0.9118 | 0.9118 | 0.9118 | 0.9118 |
| **Random Forest** | **0.9265** | **0.9143** | **0.9412** | **0.9275** |

Random Forest obtuvo el mejor desempeño general, destacando un **Recall de 94.1%** — detecta la gran mayoría de fallos reales, lo cual es crítico en mantenimiento predictivo donde un fallo no detectado puede detener la producción.

## Instalación y Ejecución

```bash
# Crear entorno virtual
python -m venv .venv

# Activar (Windows CMD)
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el pipeline completo
python main.py
```

## Dependencias

- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Autor

Proyecto desarrollado para el curso de Aprendizaje Automático — 2026.