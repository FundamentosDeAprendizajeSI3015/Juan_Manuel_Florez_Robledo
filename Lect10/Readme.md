# Proyecto de Auditoria de Clustering: Mision UdeA

Este proyecto consiste en una herramienta de auditoria diseñada para evaluar la exactitud con la que los analistas financieros de la Universidad de Antioquia clasifican los activos de la institución. Utilizando algoritmos de aprendizaje automatico, el sistema verifica si las agrupaciones sugeridas por los humanos corresponden a las estructuras de densidad y similitud real presentes en los datos.

## Estructura del Software

El sistema se compone de tres modulos principales de ejecucion:

1. clustering_pipeline.py: Es el centro de control que realiza la carga de datos, el preprocesamiento, la ejecucion de la auditoria y la generacion automatica de informes visuales.
2. subtractive_clustering.py: Un algoritmo implementado desde cero que detecta automaticamente el numero ideal de grupos segun el potencial de densidad de cada punto.
3. fuzzy_cmeans.py: Un algoritmo implementado manualmente que permite realizar clasificaciones difusas, asignando probabilidades de pertenencia en lugar de grupos rigidos.

### Preprocesamiento de Datos
Para que los algoritmos funcionen correctamente, el software realiza dos tareas criticas antes del analisis:
* Encoding de Facultades: Se aplica la tecnica One-Hot Encoding a la columna de unidades o facultades. Esto transforma las categorias de texto en variables numericas binarias, permitiendo que el modelo incluya el origen academico del activo en su calculo de similitud.
* Normalizacion: Se aplica un escalamiento estandar a todas las variables. Esto garantiza que caracteristicas con rangos grandes (como el presupuesto) no dominen sobre variables con rangos pequeños (como indicadores de riesgo), asegurando un analisis equilibrado.

## Instalacion y Preparacion

Para poner en marcha el proyecto, siga estos pasos en su terminal:

1. Cree un entorno virtual de Python: python -m venv venv
2. Active el entorno virtual:
   - En Windows: venv\Scripts\activate
   - En Linux o Mac: source venv/bin/activate
3. Instale todas las librerias necesarias: pip install -r requirements.txt

## Como realizar el Analisis

El script esta preparado para procesar dos tipos de escenarios financieros. Debe ejecutar el comando correspondiente segun el archivo que desee auditar:

Analisis del escenario ideal:
python clustering_pipeline.py --csv data/dataset_sintetico_FIRE_UdeA.csv

Analisis del escenario realista:
python clustering_pipeline.py --csv data/dataset_sintetico_FIRE_UdeA_realista.csv

## Resultados de la Mision DBSCAN

La funcion principal del sistema es auditar los activos que los analistas marcaron como Clase 1. El algoritmo DBSCAN actua como juez basandose en la concentracion de los datos:

- Precision de etiquetas: El sistema indica que porcentaje de los activos marcados por los analistas coinciden realmente con la densidad del grupo identificado por la IA.
- Identificacion de ruido: Los activos que no guardan relacion suficiente con ningun grupo son marcados como ruido. Esto revela si los analistas estan forzando clasificaciones en activos que deberian considerarse atipicos o de alto riesgo.
- Analisis de errores: El sistema identifica si los fallos ocurren en los puntos frontera, donde la similitud entre activos de distintas clases es muy alta y el juicio humano tiende a ser impreciso.

## El Bonus Track y las Probabilidades

Como valor añadido, el proyecto integra el uso de C-Medias Difuso para manejar la incertidumbre financiera. A diferencia de los metodos tradicionales, este algoritmo asigna a cada activo una probabilidad de pertenencia. 

El sistema genera automaticamente una grafica de probabilidades que muestra el grado de certidumbre:
- Si la probabilidad es cercana al 100 por ciento, la clasificacion de los analistas es muy confiable.
- Si los activos se situan cerca del 50 por ciento, existe una ambigüedad que sugiere que el activo podria pertenecer a otra categoria o requiere una revision manual profunda.

## Archivos de Salida

Al finalizar cada ejecucion, el sistema creara las carpetas resultados_ideal o resultados_realista en la raiz del proyecto. Alli encontrara:
- Mapas de calor con los perfiles promedio de cada grupo para entender su comportamiento financiero.
- Graficas comparativas que contrastan los resultados de los modelos de densidad frente a los modelos difusos.
- Histogramas de certidumbre para visualizar graficamente la confianza del sistema en las agrupaciones realizadas.