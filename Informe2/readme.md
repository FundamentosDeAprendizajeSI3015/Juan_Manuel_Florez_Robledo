# Pipeline de Mantenimiento Predictivo — AI4I 2020 (Informe 2)

Universidad EAFIT | Curso: Aprendizaje Automático | 2026-1
Autor: Juan Manuel Florez

---

## Descripcion del Proyecto

Este repositorio es la continuación directa del Informe 1. Donde antes
implementamos las fases de entendimiento, preprocesamiento y exploración
inicial, ahora añadimos tres fases avanzadas del ciclo de vida de un
sistema de aprendizaje automático:

1. Análisis no supervisado con cuatro algoritmos de clustering.
2. Reevaluación y corrección de etiquetas posiblemente incorrectas.
3. Entrenamiento y comparativa de cinco modelos supervisados sobre el
   dataset original y el dataset con etiquetas corregidas.

El dataset utilizado es el **AI4I 2020 Predictive Maintenance** de Kaggle,
que contiene 10 000 registros de telemetría de una planta de fresado
industrial, con variables de temperatura, velocidad, torque, desgaste de
herramienta y el indicador binario de fallo de máquina.

---

## Estructura del Proyecto

```text
Informe2/
├── data/
│   └── ai4i2020.csv              Dataset crudo (descargado automáticamente)
├── notebooks/
│   └── pipeline_informe2.py      Script principal del pipeline
├── outputs/
│   ├── figures/                  24 gráficas en PNG (dpi=150)
│   └── results/                  CSVs con resultados numéricos
├── presentacion/
│   └── index.html                Presentación tipo slideshow (14 slides)
├── requirements.txt              Dependencias del proyecto
└── README.md                     Este archivo
```

---

## Requisitos

- Python 3.10 o superior
- Cuenta en Kaggle con `~/.kaggle/kaggle.json` configurado
  (ir a <https://www.kaggle.com/settings> → API → Create New Token)
- Opcional: GPU NVIDIA con CUDA para acelerar Random Forest mediante cuML

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Como Ejecutar

```bash
cd notebooks
python pipeline_informe2.py
```

El script se puede ejecutar de principio a fin sin intervención manual.
Si el dataset no está descargado, lo descarga automáticamente.
Si no hay GPU CUDA disponible, el pipeline cae a CPU sin errores.

---

## Metodología

### Herencia del Informe 1

El preprocesamiento mantiene exactamente los mismos pasos del primer informe:

- Se eliminan las columnas `UDI` y `Product ID` (identificadores sin valor predictivo).
- Se imputan valores faltantes con la mediana de cada columna numérica
  (robusto frente a outliers, a diferencia de la media).
- Se eliminan filas duplicadas.
- Se aplica un filtro de consistencia física: se descartan filas con
  `Torque [Nm] <= 0`, que indican sensores con error o máquina apagada.
- Se aplica One-Hot Encoding a la variable `Type` (L/M/H quality),
  generando columnas binarias `Type_H`, `Type_L`, `Type_M`.
- Los indicadores de fallo se convierten a tipo entero.

La feature `Dif_Temperatura` (diferencia entre temperatura de proceso y
temperatura del ambiente), identificada en el Informe 1 como precursora
del fallo HDF (Heat Dissipation Failure), se analiza en el EDA.

### Analisis No Supervisado

Se aplican cuatro algoritmos de clustering sobre las features escaladas
con `StandardScaler`, sin usar la etiqueta `Machine failure`:

**K-Means**
Método del codo + Silhouette Score para elegir el k óptimo entre 2 y 10.
Se visualizan los clusters en un scatter 2D usando PCA.

**Fuzzy C-Means**
Se prueba c = 2, 3 y 4. Se elige el c con mayor FPC (Fuzzy Partition
Coefficient). A diferencia de K-Means, cada punto tiene una membresía
parcial en cada cluster, lo que resulta útil para detectar casos borderline.

**DBSCAN**
Se usa el K-distance graph para estimar el radio eps, y se explora una
grilla de (eps, min_samples). Los puntos que no pertenecen a ningún
cluster denso se clasifican como ruido (-1).

**Mean Shift**
El bandwidth se estima automáticamente. Conceptualmente es la versión
continua del Subtractive Clustering: desplaza cada punto hacia el modo
de densidad más cercano.

Para comparar los cuatro métodos se reportan: número de clusters,
Silhouette Score y Adjusted Rand Index respecto al target real
(guardado en `outputs/results/clustering_comparison.csv`).

### Reevaluacion de Etiquetas

Partiendo de que hasta el 30 % de las etiquetas en datasets industriales
pueden estar mal asignadas, se implementa un proceso de detección en dos
etapas:

**Etapa 1 — Consenso de clusters (K-Means)**
Para cada cluster se determina la etiqueta mayoritaria. Toda instancia
cuya etiqueta original difiera de la etiqueta mayoritaria de su cluster
se marca como sospechosa.

**Etapa 2 — Confident Learning (Random Forest OOF)**
Se entrena un Random Forest con validación cruzada estratificada (5-fold)
y se obtienen probabilidades out-of-fold. Son sospechosas las instancias
donde:

- `y == 1` pero `P(fallo) < 0.20` (el modelo las considera no-fallo)
- `y == 0` pero `P(fallo) > 0.80` (el modelo las considera fallo)

**Corrección conservadora (intersección)**
Solo se corrigen las etiquetas que ambos métodos marcan como sospechosas.
Usar la intersección en lugar de la unión reduce el riesgo de introducir
nuevos errores al corregir etiquetas que en realidad estaban bien.

Esto produce dos versiones del dataset:

- `dataset_A_original.csv`: etiquetas sin modificar.
- `dataset_B_corregido.csv`: etiquetas con flips en la intersección.

### Modelos Supervisados

Se entrenan cinco modelos sobre ambos datasets (A y B). Para cada uno:

1. Split 80/20 estratificado (`stratify=y`).
2. SMOTE exclusivamente en el conjunto de entrenamiento.
3. Escalado con `StandardScaler` (ajustado en train, aplicado en test).
4. Evaluación con: Accuracy, Precision, Recall, F1, ROC-AUC.

**Arbol de Decision**
GridSearchCV sobre `max_depth`, `min_samples_split` y `criterion`.
Se exporta la visualización del árbol (profundidad visual = 4) y la
importancia de features.

**Regresion Logistica**
GridSearchCV sobre `C` y `solver`. Se grafican los coeficientes por feature
(positivo = aumenta probabilidad de fallo).

**Regresion Lineal (baseline)**
Sin ajuste de hiperparámetros. Se clasifica con umbral = 0.5. Se incluye
únicamente como referencia inferior: un modelo que no supere este baseline
no justifica su complejidad.

**Random Forest**
300 árboles, sin GridSearch (el RF es relativamente robusto a sus
hiperparámetros con suficientes árboles). Si hay cuML disponible
(GPU NVIDIA), se usa la implementación acelerada por hardware.

**SVM con kernel RBF**
GridSearchCV sobre `C` y `gamma`. Requiere escalar los datos (ya hecho
en el preprocesamiento de cada dataset).

### Comparativa de Modelos

La tabla maestra (`model_comparison.csv`) consolida todas las combinaciones
modelo × dataset, ordenadas por F1 descendente. Las gráficas comparativas
incluyen:

- Bar charts de F1 y AUC agrupados por dataset.
- Heatmap de métricas normalizadas.
- Curvas ROC superpuestas para Dataset A y Dataset B por separado.

---

## Resultados Principales

### Dataset

| Parámetro              | Valor                    |
|------------------------|--------------------------|
| Total registros        | 10 000                   |
| Sin fallo (clase 0)    | 9 661 (96.61 %)          |
| Con fallo (clase 1)    | 339 (3.39 %)             |
| Desbalanceo            | 28.5 : 1                 |
| Varianza explicada PCA | 46.9 % (2 componentes)   |

Las correlaciones con `Machine failure` más altas son **Torque** (0.191)
y **Tool wear** (0.105), lo que confirma que la sobrecarga mecánica y el
desgaste de la herramienta son los principales precursores de fallo.
La diferencia de temperatura (Proceso − Ambiente) es ligeramente menor
en los casos de fallo (9.40 K) que en los normales (10.02 K), lo que
sugiere que los fallos térmicos (HDF) corresponden a condiciones donde
el sistema de disipación está al límite en lugar de superarlo.

### Clustering

| Método | Param. óptimo | Silhouette | ARI | Observación |
| --- | --- | --- | --- | --- |
| K-Means | k=3 | 0.3292 | -0.0083 | Codo claro en k=3 |
| Fuzzy C-Means | c=2 | 0.2774 | -0.0047 | FPC=0.5550 |
| DBSCAN | eps=0.3, min=10 | 0.7632 | -0.0071 | 99.6% del dataset como ruido |
| Mean Shift | bw=2.72 | 0.3292 | -0.0083 | Coincide con K-Means k=3 |

Los ARI negativos en todos los métodos indican que la estructura geométrica
del dataset no se alinea con la etiqueta de fallo: los fallos no forman un
cluster separado, sino que aparecen dispersos dentro de los clusters
normales. Esto es consistente con un dataset donde los fallos son
multimodo (HDF, TWF, PWF, OSF, RNF) y de baja prevalencia (3.4 %).

El DBSCAN con eps=0.3 y min_samples=10 obtuvo el Silhouette más alto
(0.7632), pero a costa de marcar el 99.6 % del dataset como ruido,
lo que lo hace poco útil para la corrección de etiquetas. Para esa tarea
se usó K-Means (k=3) por su cobertura completa del dataset.

### Reevaluación de Etiquetas

| Fuente                | Sospechosas detectadas |
|-----------------------|------------------------|
| Consenso K-Means      | 339 (3.39 %)           |
| Confident Learning RF | 62 (0.62 %)            |
| Intersección (flip)   | **60 (0.60 %)**        |

El método conservador (intersección) corrigió 60 etiquetas: 60 registros
etiquetados como fallo (y=1) que el Random Forest OOF consideraba no-fallo
con alta confianza (P < 0.20). Dataset A quedó con 339 fallos; Dataset B
con 279 fallos tras los flips.

### Modelos Supervisados

Resultados completos sobre el conjunto de test (20 % del dataset, 2 000 registros):

| Modelo              | Dataset | F1     | ROC-AUC | Recall |
|---------------------|---------|--------|---------|--------|
| Random Forest       | B       | 0.7040 | 0.9913  | 0.7857 |
| Árbol de Decisión   | B       | 0.6853 | 0.9277  | 0.8750 |
| Árbol de Decisión   | A       | 0.5629 | 0.8321  | 0.6912 |
| Random Forest       | A       | 0.5696 | 0.9533  | 0.6618 |
| SVM                 | A       | 0.4444 | 0.9519  | 0.7647 |
| Regresión Logística | B       | 0.2849 | 0.9422  | 0.8750 |
| Regresión Lineal    | B       | 0.2740 | 0.9435  | 0.8929 |
| Regresión Logística | A       | 0.2500 | 0.9060  | 0.8382 |
| Regresión Lineal    | A       | 0.2356 | 0.9049  | 0.7794 |

Observaciones clave:

- **La corrección de etiquetas mejoró todos los modelos.** La ganancia en
  F1 más grande la obtuvo Random Forest (+0.134) y Árbol de Decisión (+0.122).
- **El Árbol de Decisión con Dataset B** logró el Recall más alto (0.8750),
  lo que significa que detecta casi 9 de cada 10 fallos reales. Pero su
  F1 (0.6853) queda por debajo del Random Forest B (0.7040), lo que indica
  más falsos positivos.
- **Random Forest con Dataset B** es el ganador por F1 y AUC (0.9913),
  siendo el más recomendado para producción.
- La Regresión Logística y Lineal tienen AUC alto (>0.90) pero F1 bajo,
  lo que refleja que son buenos discriminadores globales pero no están
  bien calibrados para el umbral de clasificación en datos tan desbalanceados.
- La Regresión Lineal (baseline) obtuvo AUC comparable a la Logística,
  confirmando que en este dataset las dos son aproximaciones similares
  a nivel de ranking; sin embargo ninguna supera a los ensambles en F1.

### Modelo Recomendado para Producción

Random Forest entrenado sobre Dataset B (etiquetas corregidas):

- F1 = 0.7040, AUC = 0.9913, Recall = 0.7857
- Detecta 78.6 % de los fallos reales del conjunto de test.
- El AUC de 0.9913 indica que con un umbral ajustado se puede llevar el
  Recall por encima del 90 % a costa de más falsas alarmas, lo que es
  preferible en un contexto donde un fallo no detectado puede causar un
  accidente o una parada no planificada.

---

## Informe Teorico-Practico

### Problema planteado

El problema consiste en predecir si una máquina de fresado industrial
va a fallar, usando telemetría en tiempo real. Se enmarca en la definición
de aprendizaje de Tom Mitchell: el modelo aprende de la experiencia
(E = 10 000 registros etiquetados) respecto a la tarea (T = clasificar fallos)
medido con una métrica de rendimiento (P = F1-Score, con énfasis en Recall).

### Análisis no supervisado con clustering

Los cuatro métodos de clustering muestran que el espacio de features no
tiene una separación natural entre máquinas que fallan y las que no. Los
ARI negativos en todos los métodos confirman que los fallos están dispersos
dentro de los mismos clusters que la operación normal. Esto tiene sentido
dado que los cinco modos de fallo del dataset (HDF, TWF, PWF, OSF, RNF)
obedecen a mecanismos físicos distintos y sus combinaciones de features
no son necesariamente contiguas en el espacio geométrico.

K-Means con k=3 encontró tres grupos que corresponden grosso modo a tres
regímenes de operación: baja carga (Cluster 2, 1 003 puntos, 21 fallos),
carga media (Cluster 1, 6 000 puntos, 235 fallos) y carga alta (Cluster 0,
2 997 puntos, 83 fallos). Ningún cluster tiene una mayoría de fallos, lo
que explica el ARI negativo y justifica el uso del método conservador
(intersección) en la corrección de etiquetas.

### Análisis de la corrección de etiquetas

Se corrigieron 60 etiquetas (0.60 % del dataset), todas ellas registros
originalmente marcados como fallo que el Random Forest OOF predijo como
no-fallo con confianza alta. Esto redujo el conteo de fallos de 339 a 279,
lo que representa una contracción del 17.7 % en la clase positiva. La
corrección mejoró de forma consistente el F1 y el AUC de todos los modelos,
lo que indica que esas 60 instancias eran ruido que dificultaba el
aprendizaje del límite de decisión.

### Impacto de la corrección en los modelos

La corrección de etiquetas tuvo el mayor impacto sobre los modelos de
árbol (Decision Tree +12.2 pp F1, Random Forest +13.4 pp F1), que son
más sensibles al ruido en las etiquetas porque aprenden reglas duras.
Los modelos lineales (Logística y Lineal) ya tenían AUC alto sin corrección
(>0.90), lo que sugiere que el espacio de features es linealmente separable
a nivel de ranking, pero el desbalanceo extremo (28.5:1) dificulta la
calibración del umbral. SMOTE ayuda, pero no es suficiente para estos modelos
sin ajuste adicional del umbral de clasificación.

---

## Herramientas y Recursos

| Librería           | Uso                                               |
|--------------------|---------------------------------------------------|
| pandas / numpy     | Manipulación de datos                             |
| scikit-learn       | Clustering, modelos supervisados, métricas        |
| scikit-fuzzy       | Fuzzy C-Means                                     |
| imbalanced-learn   | SMOTE para manejo de desbalanceo                  |
| matplotlib/seaborn | Visualizaciones                                   |
| torch              | Detección de GPU CUDA                             |
| kaggle             | Descarga automática del dataset                   |

Dataset: [AI4I 2020 Predictive Maintenance Dataset (Kaggle)](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)

---

## Uso de Inteligencia Artificial como Herramienta de Apoyo

Este proyecto fue desarrollado con la asistencia de **Claude** (Anthropic)
como herramienta de apoyo para la generación de código y estructuración del
pipeline.

El uso de Claude se limitó a:

- Generar la estructura inicial del script .
- Sugerir mejores prácticas de código (manejo de errores, fallbacks para GPU).

Todos los criterios técnicos, decisiones metodológicas y análisis de
resultados fueron revisados y validados por el autor del proyecto.

El uso de herramientas de IA como apoyo en el proceso de aprendizaje está
alineado con las competencias de la Industria 4.0 que el curso busca
desarrollar, siempre que el estudiante comprenda y pueda explicar cada
línea del código entregado.
