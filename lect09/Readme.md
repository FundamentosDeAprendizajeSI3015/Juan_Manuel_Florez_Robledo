
# Clustering de Datos Financieros - FIRE UdeA

Este proyecto aplica algoritmos de aprendizaje no supervisado (Clustering) para analizar datasets sintéticos y realistas de información financiera universitaria. Se exploran técnicas como **K-Means** y **DBSCAN** para identificar patrones y agrupamientos naturales en los datos.

## 1. Estructura del Proyecto

* `data/`: Contiene los archivos CSV (`dataset_sintetico_FIRE_UdeA.csv` y su versión realista).
* `ejAgrupamiento_kmeans_dbscan.ipynb`: Cuaderno de Jupyter con el análisis y visualizaciones.
* `requirements.txt`: Lista de dependencias necesarias.
* `.gitignore`: Configuración para excluir archivos innecesarios (como el entorno virtual).

## 2. Instalación y Requisitos

Sigue estos pasos para configurar el entorno localmente:

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DE_TU_REPOSITORIO>
   cd <NOMBRE_DE_LA_CARPETA>
   ```

2. **Crear y activar entorno virtual:**
   ```bash
   python -m venv env
   # En Windows:
   .\env\Scripts\activate
   # En macOS/Linux:
   source env/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## 3. Uso

Ejecuta el cuaderno de Jupyter para ver el análisis de clustering:
1. Abre VS Code o tu terminal.
2. Inicia el servidor de Jupyter: `jupyter notebook`.
3. Abre el archivo `ejAgrupamiento_kmeans_dbscan.ipynb` y ejecuta todas las celdas.

## 4. Algoritmos Implementados
* **K-Means:** Determinado el valor óptimo de K mediante el Método del Codo.
* **DBSCAN:** Utilizado para detección de clusters basados en densidad y manejo de ruido.

---
**Autor:** Juan Manuel Florez Robledo - Estudiante de Ingeniería de Sistemas
```
