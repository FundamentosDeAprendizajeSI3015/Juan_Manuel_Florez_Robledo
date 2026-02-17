import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from mpl_toolkits.mplot3d import Axes3D

# 1. Cargar los datos ya procesados en el pipeline anterior
def cargar_datos():
    ruta = "data/datos_procesados/ai4i2020_procesado.csv"
    return pd.read_csv(ruta)

# 2. Generar UMAP 3D (Reducción de dimensionalidad)
def graficar_umap_3d(df):
    print("⏳ Calculando UMAP 3D (esto puede tardar unos segundos)...")
    
    # Seleccionamos solo las características numéricas para el análisis
    features = df.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    
    # Configuramos UMAP para 3 dimensiones
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedding = reducer.fit_transform(features)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficamos: puntos azules son normales, rojos son fallos
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=df['Machine failure'], cmap='coolwarm', s=2, alpha=0.6
    )
    
    ax.set_title('Visualización UMAP 3D de Fallos de Motor')
    plt.colorbar(scatter, label='Fallo (1) vs Normal (0)')
    plt.savefig("data/datos_procesados/umap_3d_fallos.png")
    print("✅ Gráfica UMAP 3D guardada.")

# 3. Mapa de Calor de Correlación (Semana 4 - Exploración)
def graficar_correlacion(df):
    plt.figure(figsize=(12, 10))
    # Calculamos qué tanto se relaciona cada variable con el fallo
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='RdYlGn', fmt=".2f")
    plt.title('Mapa de Calor: ¿Qué variables causan el fallo?')
    plt.savefig("data/datos_procesados/mapa_calor_correlacion.png")
    print("✅ Mapa de calor guardado.")

# 4. Boxplot de la característica que creamos (Dif_Temperatura)
def graficar_feature_engineered(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Machine failure', y='Dif_Temperatura', data=df)
    plt.title('Impacto de la Diferencia de Temperatura en los Fallos')
    plt.savefig("data/datos_procesados/boxplot_temperatura.png")
    print("✅ Boxplot de temperatura guardado.")

if __name__ == "__main__":
    df = cargar_datos()
    graficar_umap_3d(df)
    graficar_correlacion(df)
    graficar_feature_engineered(df)
    print("\n🚀 Todas las visualizaciones están listas en la carpeta 'data/datos_procesados/'")