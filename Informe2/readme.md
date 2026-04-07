```markdown
# Mantenimiento Predictivo Inteligente: Auditoria de Datos y Clasificacion
**Universidad EAFIT | Sistemas de Aprendizaje Automatico | Informe 2**
**Autor:** Juan Manuel Florez

Este repositorio contiene la implementacion completa de un pipeline avanzado de Machine Learning aplicado al dataset industrial AI4I 2020 Predictive Maintenance. El objetivo principal de esta investigacion es demostrar el impacto del principio GIGO (Garbage In, Garbage Out) en entornos industriales, utilizando el Aprendizaje No Supervisado como mecanismo de auditoria para descubrir y purificar ruido en las etiquetas (Ground Truth) antes del entrenamiento de modelos predictivos.

## Resumen del Proyecto y Definicion TPE

Basado en el framework de Tom Mitchell:
* **Tarea (T):** Clasificacion binaria para predecir fallas mecanicas preventivas.
* **Rendimiento (P):** Maximizacion del F1-Score y Recall, minimizando los Falsos Negativos (fallas no detectadas que causan paradas de planta).
* **Experiencia (E):** Dataset historico de 10,000 registros telemetricos de una maquina de fresado.

El desafio critico resuelto en este proyecto fue el desbalance extremo de clases (28.5:1, con solo 3.39% de fallas) combinado con errores de etiquetado humano.

## Resultados Clave de la Investigacion

La hipotesis inicial sugeria que el dataset contenia fallas ocultas. Sin embargo, la auditoria de datos revelo el problema opuesto: la existencia de "falsas alarmas". 

Al cruzar la topologia fisica descubierta mediante algoritmos de clustering con las probabilidades de un modelo base (Out-of-Fold Confident Learning), se descubrieron 60 registros (0.60% del dataset) que estaban etiquetados como "Falla", pero cuyos perfiles termodinamicos y de torque eran identicos a la operacion normal.

Purificar estos 60 registros genero un salto cuantitativo en la capacidad predictiva de todas las arquitecturas supervisadas evaluadas:
* **Random Forest:** Incremento su F1-Score de 0.5696 (sobre datos sucios) a 0.7040 (sobre datos limpios), una mejora absoluta del 13.44%.
* **Capacidad de Discriminacion:** El area bajo la curva (ROC-AUC) alcanzo un 0.9913.
* **Deteccion de Fallas Reales:** El Recall subio al 78.57%, garantizando un sistema robusto para produccion.

## Arquitectura del Pipeline

### 1. Preprocesamiento e Ingenieria de Caracteristicas (EDA)
* Imputacion de valores faltantes mediante la mediana para evitar la sensibilidad a valores atipicos extremos.
* Eliminacion de columnas identificadoras que no aportan varianza (UDI, Product ID).
* Creacion de la variable `temp_diff` (Diferencia entre la temperatura del proceso y la del aire), la cual demostro ser uno de los mejores predictores termicos de fallas HDF (Heat Dissipation Failure).

### 2. Auditoria No Supervisada (Clustering)
Se evaluo una bateria de modelos para descubrir la estructura natural de los datos sin sesgo de etiquetas:
* **K-Means (k=3):** El modelo ganador (Silhouette = 0.3292). Logro identificar tres estados operativos fisicos: Operacion Estable, Desgaste Progresivo y Falla Critica.
* **DBSCAN:** Utilizado para la deteccion de anomalias puras, aislando el ruido absoluto del dataset.
* **Fuzzy C-Means & Mean Shift:** Utilizados para corroborar las fronteras de decision difusas y los centros de masa de la operacion de la planta.

### 3. Reevaluacion de Etiquetas (Label Cleaning)
Se genero un "Dataset B" corregido mediante una politica de consenso estricto: solo se modifico una etiqueta si el analisis fisico (K-Means) y el analisis probabilistico (Random Forest) coincidian en que se trataba de un falso positivo humano.

### 4. Benchmark Supervisado
Se evaluaron diversos modelos aplicando SMOTE (Synthetic Minority Over-sampling Technique) unicamente en los conjuntos de entrenamiento para mitigar el desbalance del 3.39%:
* **Arboles de Decision (CART):** Alta capacidad para aislar umbrales de desgaste no lineales.
* **Regresion Logistica:** Modelo base probabilistico (con aplicacion previa de StandardScaler).
* **Regresion Lineal:** Incluido estrictamente como baseline negativo teorico, demostrando matematicamente la ineficiencia de los modelos sin limite [0,1] para clasificacion binaria de alta varianza.
* **Random Forest & SVM:** Arquitecturas de alta complejidad para extraer el maximo rendimiento del dataset purificado.

## Estructura del Repositorio

```text
Informe2/
|-- data/
|   |-- ai4i2020.csv                    # Dataset original
|-- notebooks/
|   |-- pipeline_informe2.py            # Script maestro (Clustering, Limpieza, Evaluacion)
|-- outputs/
|   |-- figures/                        # 24 graficas generadas automaticamente (PCA, Codos, ROC, Heatmaps)
|   |-- results/
|       |-- clustering_comparison.csv   # Metricas de algoritmos no supervisados
|       |-- dataset_A_original.csv      # Dataset preprocesado con ruido
|       |-- dataset_B_corregido.csv     # Dataset purificado (Ground Truth ajustado)
|       |-- model_comparison.csv        # Benchmark final de modelos supervisados
|-- presentacion/
|   |-- index.html                      # Presentacion ejecutiva interactiva que consume las graficas
|-- requirements.txt                    # Dependencias estrictas del proyecto
|-- README.md                           # Documentacion principal
```

## Instrucciones de Reproduccion

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/tu-usuario/mantenimiento-predictivo-eafit.git](https://github.com/tu-usuario/mantenimiento-predictivo-eafit.git)
   cd mantenimiento-predictivo-eafit
   ```

2. **Crear entorno virtual e instalar dependencias:**
   Se recomienda usar Python 3.8 o superior.
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows usar: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Ejecutar el Pipeline de Analisis:**
   Este script ejecutara todas las fases matematicas, entrenara los modelos, exportara los CSV de metricas y generara 24 graficas en alta resolucion en el directorio `outputs/figures/`.
   ```bash
   cd notebooks
   python pipeline_informe2.py
   ```

4. **Visualizar el Informe Interactivo:**
   Navegue a la carpeta `presentacion` y abra el archivo `index.html` en cualquier navegador web moderno. Presione `F11` para ver la defensa grafica de los resultados en modo de pantalla completa.

## Tecnologias Utilizadas

* **Lenguaje:** Python 3.x
* **Procesamiento de Datos:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (K-Means, DBSCAN, Random Forest, SVM, Regresion Logistica/Lineal, PCA)
* **Balanceo de Datos:** Imbalanced-Learn (SMOTE)
* **Logica Difusa:** Scikit-Fuzzy (C-Means)
* **Visualizacion Cientifica:** Matplotlib, Seaborn
* **Presentacion:** HTML5, CSS3, JS Vanilla (Integracion estatica sin dependencias web)
```