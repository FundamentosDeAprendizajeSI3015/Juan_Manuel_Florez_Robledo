import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Manejo de dependencia para Binary Encoding
try:
    from category_encoders import BinaryEncoder
except ImportError:
    print("[!] Instalando 'category_encoders'...")
    os.system('pip install category_encoders')
    from category_encoders import BinaryEncoder

def ejecutar_laboratorio():
    # 1. PREPARACIÓN DE CARPETAS
    if not os.path.exists('graficas'): os.makedirs('graficas')
    if not os.path.exists('salidas'): os.makedirs('salidas')

    # 2. CARGA DE DATOS
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    # 3. ANÁLISIS ESTADÍSTICO (Tendencia Central, Dispersión y Posición)
    # Medidas de Tendencia Central
    mean_age = df['Age'].mean()
    median_age = df['Age'].median()
    mode_fare = df['Fare'].mode()[0]

    # Medidas de Dispersión
    std_fare = df['Fare'].std()
    var_fare = df['Fare'].var()

    # Medidas de Posición (Cuartiles)
    cuartiles_fare = df['Fare'].quantile([0.25, 0.5, 0.75])
    
    # 4. EXPLORACIÓN GRÁFICA Y ELIMINACIÓN DE OUTLIERS
    # Boxplot (Posición y Outliers)
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['Fare'], color='salmon')
    plt.title('Detección de Outliers en Tarifa (Fare)')
    plt.savefig('graficas/01_boxplot_fare.png')
    plt.close()

    # Eliminación de Outliers mediante IQR
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    limite_sup = Q3 + 1.5 * IQR
    df_clean = df[df['Fare'] <= limite_sup].copy()
    outliers_eliminados = len(df) - len(df_clean)

    # Histogramas (Distribución)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df_clean['Age'].dropna(), kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Distribución de Edad (Histograma)')
    sns.histplot(df_clean['Fare'], kde=True, ax=axes[1], color='teal')
    axes[1].set_title('Distribución de Tarifa (Histograma)')
    plt.savefig('graficas/02_histogramas.png')
    plt.close()

    # Gráfico de Dispersión (Relación entre variables)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_clean, x='Age', y='Fare', hue='Survived', palette='viridis', alpha=0.7)
    plt.title('Relación Edad vs Tarifa (Dispersión)')
    plt.savefig('graficas/03_dispersion_edad_fare.png')
    plt.close()

    # 5. TRANSFORMACIÓN DE COLUMNAS
    # Label Encoding (Sex)
    le = LabelEncoder()
    df_clean['Sex_Encoded'] = le.fit_transform(df_clean['Sex'])

    # Binary Encoding (Pclass)
    be = BinaryEncoder(cols=['Pclass'])
    df_clean = be.fit_transform(df_clean)

    # One Hot Encoding (Embarked)
    df_clean = pd.get_dummies(df_clean, columns=['Embarked'], prefix='Port')

    # Escalado (StandardScaler para Age, Min-Max para Fare)
    scaler_std = StandardScaler()
    df_clean['Age_Scaled'] = scaler_std.fit_transform(df_clean[['Age']].fillna(df_clean['Age'].median()))

    scaler_mm = MinMaxScaler()
    df_clean['Fare_Scaled'] = scaler_mm.fit_transform(df_clean[['Fare']])

    # Transformación Logarítmica
    df_clean['Fare_Log'] = np.log1p(df_clean['Fare'])

    # 6. CORRELACIÓN Y ELIMINACIÓN
    plt.figure(figsize=(12, 10))
    matriz_corr = df_clean.corr(numeric_only=True)
    sns.heatmap(matriz_corr, annot=True, cmap='RdBu_r', fmt=".2f")
    plt.title('Matriz de Correlación')
    plt.savefig('graficas/04_matriz_correlacion.png')
    plt.close()

    # Eliminación de columnas redundantes o no útiles para el modelo
    df_final = df_clean.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'Age', 'Fare'])

    # 7. GENERACIÓN DE ARCHIVOS DE SALIDA (CSV y Markdown)
    # Guardar CSV procesado
    df_final.to_csv('salidas/titanic_limpio_2026.csv', index=False)

    # Guardar Reporte Markdown
    with open('salidas/informe_EDA.md', 'w', encoding='utf-8') as m:
        m.write("# Informe de Exploración de Datos (EDA) - Titanic 2026\n\n")
        m.write("## 1. Medidas Estadísticas\n")
        m.write(f"* **Media de Edad:** {mean_age:.2f}\n")
        m.write(f"* **Mediana de Edad:** {median_age:.2f}\n")
        m.write(f"* **Desviación Estándar (Fare):** {std_fare:.2f}\n")
        m.write(f"* **Varianza (Fare):** {var_fare:.2f}\n\n")
        
        m.write("## 2. Análisis de Outliers\n")
        m.write(f"* **Límite superior calculado (IQR):** {limite_sup:.2f}\n")
        m.write(f"* **Registros eliminados:** {outliers_eliminados}\n\n")
        
        m.write("## 3. Conclusiones del Laboratorio\n")
        m.write("1. **Distribución:** La edad sigue una distribución aproximadamente normal centrada en los 28-30 años.\n")
        m.write("2. **Correlación:** Se observó una correlación negativa entre la clase y la supervivencia; pasajeros en clases superiores tuvieron mayor probabilidad de vivir.\n")
        m.write("3. **Transformaciones:** Se aplicó Binary Encoding a 'Pclass' para evitar la jerarquía artificial y One-Hot a los puertos de embarque.\n")
        m.write("4. **Escalado:** Se estandarizó la edad para facilitar la convergencia de modelos de ML.")

    print("\n=== PROCESO FINALIZADO CON ÉXITO ===")
    print(f"- Dataset procesado: salidas/titanic_limpio_2026.csv")
    print(f"- Reporte generado: salidas/informe_EDA.md")
    print(f"- Gráficas guardadas en: /graficas")

if __name__ == "__main__":
    ejecutar_laboratorio()