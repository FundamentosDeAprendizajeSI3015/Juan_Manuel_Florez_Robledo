
---

# FIRE-UdeA 

**Financial Institutional Risk Estimator — Universidad de Antioquia**

Sistema predictivo de riesgo financiero basado en Machine Learning para estimar la probabilidad de tensión de caja ($t+1$) en unidades académicas y administrativas de la UdeA.

---

##  Definición del Problema (Framework Mitchell)

* **Tarea (T):** Clasificar si una unidad presentará tensión financiera (cash stress) en el periodo $t+1$.
* **Rendimiento (P):** ROC-AUC, F1-Score y Brier Score.
* **Experiencia (E):** Datasets históricos con indicadores financieros (Sintético: 500 registros | Realista Panel: 80 registros).
* **Categoría:** Aprendizaje Supervisado — Clasificación Binaria.

---

##  Resultados Actuales (Última Ejecución)

El pipeline evalúa dos escenarios: un dataset general balanceado y un dataset realista con estructura de panel temporal (time-series split).

| Métrica | Dataset Sintético (500) | Dataset Realista (80) |
| --- | --- | --- |
| **Mejor Modelo** | **Decision Tree** | **Logistic Regression** |
| **ROC-AUC** | 1.0000 | 0.8958 |
| **F1-Score** | 1.0000 | 0.9231 |
| **Accuracy** | 100% | 87.5% |
| **Brier Score** | 0.0000 | 0.1653 |

### Desempeño por Unidad (Dataset Realista)

El modelo logra una alta sensibilidad, identificando correctamente la tensión en unidades críticas como el **Nivel Central**, **Sedes** e **Ingeniería**.

---

## 🛠️ Estructura del Proyecto

```text
FIRE-UdeA/
├── data/
│   ├── raw/                # Datasets originales
│   └── processed/          # Features procesados y datasets limpios
├── src/                    # Código fuente modular (loader, features, model)
├── models/                 # Modelos serializados (.pkl / .joblib)
├── outputs/                # Visualizaciones (Confusion Matrix, Feature Importance)
├── requirements.txt        # Dependencias del proyecto
└── main.py                 # Orquestador del pipeline completo

```

---

##  Quickstart

### Opción A: Script automático (Recomendado)

```bash
git clone https://github.com/JuanmaFl/FIRE-UdeA.git
cd FIRE-UdeA
bash setup_env.sh

```

### Opción B: Ejecución del Pipeline

Una vez activado el entorno virtual (`.venv`), ejecuta el análisis completo:

```bash
python main.py

```

---

##  Ingeniería de Características (Features)

El modelo utiliza una combinación de indicadores financieros base y variables derivadas:

* **Liquidez e Insolvencia:** Ratio corriente y días de efectivo.
* **Concentración:** Índice Herfindahl (HHI) de fuentes de ingreso.
* **Estructura de Costos:** `gp_ratio` (Gasto personal / Ingresos).
* **Dinámica Temporal:** Lags ($t-1$), diferencias interanuales y medias móviles de 2 años (solo en modelo realista).

---

##  Stack Tecnológico

* **Lenguaje:** Python 3.12
* **ML:** Scikit-learn, Gradient Boosting, SHAP para interpretabilidad.
* **Datos:** Pandas (procesamiento de paneles temporales), NumPy.
* **Infraestructura:** Diseño orientado a Azure ML Service & ADLS Gen2.

---

##  Autor

**Juan Manuel Flórez** — [@JuanmaFl](https://github.com/JuanmaFl)

Ingeniería de Sistemas, 7° semestre — Universidad EAFIT

---

¿Te gustaría que profundice en la explicación de por qué la **Regresión Logística** superó al **Gradient Boosting** en el dataset realista (posiblemente por el tamaño de la muestra)?