# Implementacion de Regresion Lineal y Logistica - Lectura 5

Este repositorio contiene la solucion tecnica al taller de la Lectura 5 de la asignatura Fundamentos de Aprendizaje Automatico (2026). El objetivo es aplicar modelos de regresion y clasificacion utilizando la libreria Scikit-Learn sobre el conjunto de datos del Titanic.

## Estructura del Repositorio

* **regresion_lineal.py**: Script en Python que aborda la prediccion de una variable continua (Edad).
* **regresion_logistica.py**: Script en Python que aborda la clasificacion binaria (Supervivencia).
* **outputs/**: Directorio generado automaticamente que contiene los reportes de metricas y las visualizaciones en formato PNG y TXT.
* **requirements.txt**: Archivo con las dependencias necesarias para ejecutar el proyecto.
* **.gitignore**: Configuracion para evitar la carga de archivos temporales y entornos virtuales.

## Requisitos Previos

Es necesario contar con Python 3.8 o superior. Se recomienda el uso de un entorno virtual para aislar las dependencias del sistema.

### Instalacion

1. Crear el entorno virtual:
   python -m venv .venv

2. Activar el entorno:
   Windows: .venv\Scripts\activate
   Linux/Mac: source .venv/bin/activate

3. Instalar librerias:
   pip install -r requirements.txt

## Ejecucion

Para generar los resultados y las graficas, ejecute los scripts desde la raiz del proyecto:

python regresion_lineal.py
python regresion_logistica.py

## Resumen de Actividades Realizadas

### 1. Regresion Lineal (Prediccion de Edad)
Se selecciono la columna age como variable objetivo. Se implemento un flujo de trabajo que incluye:
* Division de datos en entrenamiento y prueba (80/20).
* Grafica de dispersion inicial diferenciando los conjuntos por colores.
* Creacion de Pipelines con escalado estandar y expansion polinomica.
* Entrenamiento de modelos con regularizacion Ridge y Lasso.
* Busqueda de hiperparametros mediante RandomizedSearchCV con validacion cruzada.
* Calculo de metricas R2 y Error Absoluto Medio (MAE).
* Grafica de dispersion comparativa entre valores reales y predichos.



### 2. Regresion Logistica (Clasificacion de Supervivencia)
Se utilizo la columna survived para realizar una clasificacion basada en caracteristicas socioeconomicas. El proceso incluyo:
* Limpieza y transformacion de datos (imputacion de nulos y mapeo de variables categoricas).
* Definicion de Pipeline con escalado y expansion polinomica de grado 2.
* Optimizacion del parametro de regularizacion C y seleccion del solver.
* Evaluacion mediante Accuracy y F1-Score.
* Generacion de la Matriz de Confusion para validar el rendimiento del clasificador.



## Notas sobre los Resultados

Segun las pruebas ejecutadas, el modelo de regresion logistica alcanza una precision cercana al 82%, demostrando una buena capacidad predictiva para la supervivencia de los pasajeros. En la regresion lineal, el Error Absoluto Medio (MAE) se situa alrededor de los 9.4 años, lo que sugiere que las variables disponibles explican parcialmente la variabilidad de la edad.

Los avisos de convergencia (ConvergenceWarning) que pueden aparecer durante la ejecucion de la regresion lineal son esperados. Esto ocurre cuando el optimizador de Lasso requiere mas iteraciones de las predefinidas para converger en ciertas combinaciones de hiperparametros aleatorios. Los resultados almacenados en la carpeta de salidas son validos para los fines academicos de este taller.