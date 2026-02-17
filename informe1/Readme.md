# Pipeline de Mantenimiento Predictivo Inteligente (AI4I 2020)

Este proyecto implementa las primeras tres fases del ciclo de vida de un sistema de **Aprendizaje Automático**, aplicando los fundamentos de preprocesamiento, limpieza y análisis de casos de estudio industriales para una planta de fresado.

## 1. Descripción del Dataset
El pipeline utiliza el dataset **AI4I 2020 Predictive Maintenance** de Kaggle. Consiste en 10,000 registros con 14 características técnicas que monitorean el estado de una máquina:
* **Temperaturas:** Aire y Proceso (en Kelvin, $K$).
* **Métricas de operación:** Velocidad de rotación ($rpm$), Torque ($Nm$) y Desgaste de herramienta ($min$).
* **Etiqueta objetivo:** `Machine failure` (indica si la máquina falló por cualquier causa).

---

## 2. Fase 1: Entendimiento y Definición (Semana 2)
En esta fase inicial, definimos el marco del problema basado en la notación de aprendizaje estadístico para asegurar que el modelo tenga un impacto real en el negocio.

* **Definición Formal ($T, P, E$):** Según Tom Mitchell, el programa aprende de la experiencia ($E$) con respecto a una tarea ($T$) y una métrica de rendimiento ($P$):
    * **Tarea ($T$):** Clasificar si una instancia indica un fallo de la máquina.
    * **Rendimiento ($P$):** Precisión de la clasificación (porcentaje de fallos detectados correctamente).
    * **Experiencia ($E$):** Dataset de 10,000 ejemplos de entrenamiento etiquetados.
* **Categoría de ML:** Se define como **Aprendizaje Supervisado**, ya que utilizamos datos con salidas deseadas u objetivos.
* **Contexto Interdisciplinario:** Para evitar errores como los de la Lectura 2 (donde se ignoró el ciclo de descongelamiento de los motores), este pipeline reconoce que picos de temperatura pueden ser normales o fallos específicos como el **HDF** (Heat Dissipation Failure).



---

## 3. Fase 2: Preprocesamiento y Limpieza (Semana 3)
El preprocesamiento es la fase más crítica, ocupando entre el **70% y el 80% del tiempo**. Se rige bajo el principio **GIGO** (*Garbage In, Garbage Out*): si entra basura, sale basura.

* **Carga y Formato:** Uso de la librería **Pandas** para cargar el CSV (`pd.read_csv`).
* **Manejo de Valores Faltantes ($NaNs$):**
    * Detección de nulos mediante `df.isnull().sum()`.
    * **Imputación inteligente:** Uso de la mediana para las temperaturas (`df.fillna`) para evitar que valores atípicos afecten la robustez del modelo.
* **Manipulación de Estructura:**
    * **Eliminación de columnas:** Se descartan `UDI` y `Product ID` con `df.drop`, ya que son identificadores únicos que no aportan patrones reales.
    * **Consistencia:** Eliminación de filas duplicadas mediante `df.drop_duplicates` para evitar redundancias.



---

## 4. Fase 3: Exploración y Transformación (Semana 4)
Preparamos las características ($Features$) para que los algoritmos puedan procesarlas y validamos la lógica de los datos.

* **Inspección Rápida:**
    * `df.info()`: Verifica tipos de datos y memoria usada.
    * `df.describe()`: Obtiene estadísticas descriptivas (media, min, max, percentiles).
* **Transformación de Datos:**
    * **One-Hot Encoding:** La variable categórica `Type` (L, M, H) se transforma en columnas binarias (`pd.get_dummies`) para que el modelo comprenda la calidad del producto.
    * **Conversión de Tipos:** Aseguramos que los indicadores de fallo sean reconocidos como tipos enteros o booleanos usando `astype`.
* **Ingeniería de Características ($Feature Engineering$):**
    * **Etiquetado Contextual:** Siguiendo la solución de la Lectura 2, se crean variables que expliquen comportamientos normales de la planta.
    * **Validación Lógica:** Se verifica la diferencia de temperatura ($Process - Air$), ya que valores específicos son precursores de fallas térmicas.



---

## 5. Conclusión
Este pipeline garantiza una base sólida para el sistema de mantenimiento predictivo. Al enfocarse en las primeras tres etapas del ciclo de vida, se asegura que el modelo no sea una "caja negra", sino una herramienta interpretable y robusta para la industria 4.0.
