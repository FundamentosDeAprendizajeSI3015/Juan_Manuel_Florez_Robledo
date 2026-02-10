# Informe de Exploración de Datos (EDA) - Titanic 2026
**Asignatura:** SI3015 - Fundamentos de Aprendizaje Automático  
**Dataset:** Titanic (Kaggle)  
**Estado:** Procesado y Limpio

---

## 1. Análisis de Medidas Estadísticas
El análisis inicial de las variables numéricas permite entender la naturaleza y el comportamiento de la tripulación antes de cualquier procesamiento.

| Medida | Edad (Age) | Tarifa (Fare) |
| :--- | :--- | :--- |
| **Media** | 29.70 | 32.20 |
| **Mediana** | 28.00 | 14.45 |
| **Desviación Estándar** | 14.53 | 49.69 |
| **Varianza** | 211.02 | 2469.44 |

### Interpretación de resultados:
* **Edad:** La media y la mediana son cercanas, indicando una distribución con poco sesgo, lo que facilita la aplicación de **StandardScaler**.
* **Tarifa:** La varianza es extremadamente alta, lo que evidencia una disparidad económica masiva y la presencia de valores atípicos que requieren limpieza.



---

## 2. Medidas de Posición y Gestión de Outliers
Se utilizó el método del **Rango Intercuartílico (IQR)** para identificar y mitigar el impacto de los valores atípicos en la variable *Fare*.

* **Q1 (25%):** 7.91
* **Q2 (Mediana):** 14.45
* **Q3 (75%):** 31.00
* **Límite Superior Calculado:** 65.63

**Acción realizada:** Se eliminaron **116 registros** que superaban el límite superior. Esta limpieza previene que el modelo de aprendizaje automático sea sesgado por valores de "lujo" que no representan el patrón general de supervivencia.



---

## 3. Relación entre Variables (Análisis de Dispersión)
Al analizar la relación entre **Edad** y **Tarifa** mediante gráficos de dispersión, se concluye:
1.  Los pasajeros de **Primera Clase** (tarifas altas) presentan una tasa de supervivencia visualmente superior en comparación con la Tercera Clase.
2.  No existe una correlación lineal directa entre edad y tarifa, pero los rangos de edad infantiles muestran una mayor densidad de supervivencia, validando la política histórica de rescate.

---

## 4. Transformación de Columnas (Feature Engineering)
Para cumplir con los requisitos del laboratorio, se aplicaron las siguientes técnicas de codificación:

* **Label Encoding:** Aplicado a `Sex`. Convirtió categorías textuales en valores binarios (0 y 1).
* **Binary Encoding:** Aplicado a `Pclass`. Representa la clase en formato de bits, optimizando el espacio dimensional y evitando jerarquías artificiales complezas.
* **One-Hot Encoding:** Aplicado a `Embarked`. Creó columnas booleanas para cada puerto (Port_C, Port_Q, Port_S), eliminando cualquier orden intrínseco inexistente entre ciudades.
* **Transformación Logarítmica:** Aplicada sobre `Fare` para normalizar el sesgo residual, permitiendo que la distribución sea más "amigable" para modelos lineales.



---

## 5. Escalado de Características
Se implementaron dos estrategias de escalado para normalizar las magnitudes de las variables:
1.  **StandardScaler (Edad):** Centró la variable en media 0 y desviación 1, ideal para algoritmos que asumen distribuciones gaussianas.
2.  **Min-Max Scaling (Tarifa):** Comprimió los valores al rango $[0, 1]$, asegurando que el costo del boleto no domine sobre otras variables por su valor nominal.

---

## 6. Análisis de Correlación Final
La matriz de correlación de Pearson obtenida post-procesamiento confirma que:
* Existe una **fuerte correlación negativa** entre la Clase (`Pclass`) y la Supervivencia (`Survived`).
* El género codificado (`Sex_Label`) resultó ser la característica con mayor peso predictivo individual en este conjunto de datos.



---

## 7. Conclusiones Generales
1.  **Limpieza:** El tratamiento de outliers mediante IQR fue crítico, ya que redujo el ruido en la variable *Fare*, permitiendo una mejor visualización de la relación entre costo y clase.
2.  **Transformación:** La combinación de diferentes métodos de Encoding asegura que los datos categóricos mantengan su valor informativo sin introducir sesgos numéricos.
3.  **Preparación:** Tras el escalado y la transformación logarítmica, el dataset está técnicamente optimizado para alimentar algoritmos de clasificación como **Random Forest** o **Redes Neuronales**.

---
*Generado automáticamente por el script de análisis EDA - 2026*