import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
import umap.umap_ as umap
from subtractive_clustering import subtractive_clustering
from fuzzy_cmeans import fuzzy_c_means

# Estilo visual
plt.style.use('dark_background')

def create_folders():
    for folder in ['resultados_ideal', 'resultados_realista']:
        if not os.path.exists(folder):
            os.makedirs(folder)

def plot_membership_probabilities(u_matrix, output_path):
    plt.figure(figsize=(10, 6))
    max_probs = np.max(u_matrix, axis=0)
    sns.histplot(max_probs, kde=True, color="cyan", bins=30)
    plt.title("Certidumbre de Pertenencia (Fuzzy C-Means)")
    plt.xlabel("Probabilidad Máxima")
    plt.ylabel("Frecuencia")
    plt.savefig(f"{output_path}/grafica_probabilidades_fuzzy.png")
    plt.close()

def plot_feature_profiles(df_numeric, labels, output_path, title):
    df_temp = df_numeric.copy()
    df_temp['Cluster'] = labels
    profile = df_temp[df_temp['Cluster'] != -1].groupby('Cluster').mean()
    
    if profile.empty:
        return

    plt.figure(figsize=(12, 8))
    sns.heatmap(profile, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(f"Perfil de Variables por Cluster: {title}")
    plt.savefig(f"{output_path}/perfil_{title.lower().replace(' ', '_')}.png")
    plt.close()

def run_analysis(csv_path, n_clusters, ra, rb):
    is_realista = "realista" in csv_path
    folder = "resultados_realista" if is_realista else "resultados_ideal"
    eps_val = 4.8 if is_realista else 2.3 

    # 1. CARGA Y ENCODING
    df = pd.read_csv(csv_path).dropna()
    
    # Identificar columnas para el modelo
    # 'label' y 'anio' se excluyen del entrenamiento
    y_true = df['label']
    
    # Aplicar One-Hot Encoding a la facultad ('unidad')
    X_raw = df.drop(columns=['label', 'anio'], errors='ignore')
    if 'unidad' in X_raw.columns:
        X_encoded = pd.get_dummies(X_raw, columns=['unidad'], prefix='facultad')
    else:
        X_encoded = X_raw

    # 2. NORMALIZACIÓN (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    print(f"\n🚀 --- ANALIZANDO: {csv_path} ---")
    print(f"Dimensiones tras encoding: {X_encoded.shape}")

    # 3. EJECUCIÓN ALGORITMOS
    # DBSCAN (Misión)
    db = DBSCAN(eps=eps_val, min_samples=10).fit(X_scaled)
    db_labels = db.labels_

    # Bonus Track (Scratch)
    centers = subtractive_clustering(X_scaled, ra=ra, rb=rb)
    fcm_centers, u_matrix = fuzzy_c_means(X_scaled, n_clusters=n_clusters, init_centers=centers)
    fcm_labels = np.argmax(u_matrix, axis=0)

    # 4. CÁLCULO DE PORCENTAJES DE LA MISIÓN
    df['dbscan_res'] = db_labels
    clase_1_humana = df[df['label'] == 1]
    total_1 = len(clase_1_humana)
    
    if total_1 > 0:
        real_c1 = (clase_1_humana['dbscan_res'] == 0).sum()
        real_c2 = (clase_1_humana['dbscan_res'] == 1).sum()
        ruido = (clase_1_humana['dbscan_res'] == -1).sum()

        print(f"📊 RESULTADOS MISIÓN DBSCAN ({folder.upper()}):")
        print(f"   - ✅ Realmente Clase 1: {(real_c1/total_1)*100:.2f}%")
        print(f"   - ⚠️ Realmente Clase 2: {(real_c2/total_1)*100:.2f}%")
        print(f"   - 🚫 Outliers (Ruido): {(ruido/total_1)*100:.2f}%")

    # 5. VISUALIZACIÓN UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    plot_membership_probabilities(u_matrix, folder)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(embedding[:,0], embedding[:,1], c=db_labels, cmap='Spectral', s=15)
    plt.title(f"DBSCAN (Misión) - {folder}")
    
    plt.subplot(1, 2, 2)
    plt.scatter(embedding[:,0], embedding[:,1], c=fcm_labels, cmap='coolwarm', s=15)
    plt.title(f"Fuzzy C-Means (Bonus) - {folder}")
    
    plt.savefig(f"{folder}/comparativa_final.png")
    plt.close()

    # Perfiles (Usamos solo variables numéricas originales para el Heatmap para que sea legible)
    X_numeric_only = X_raw.select_dtypes(include=[np.number])
    plot_feature_profiles(X_numeric_only, db_labels, folder, "DBSCAN")
    plot_feature_profiles(X_numeric_only, fcm_labels, folder, "Fuzzy C-Means")

    print(f"✅ Reporte completo en ./{folder}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--n_clusters', type=int, default=2)
    parser.add_argument('--ra', type=float, default=0.5)
    parser.add_argument('--rb', type=float, default=0.75)
    args = parser.parse_args()

    create_folders()
    run_analysis(args.csv, args.n_clusters, args.ra, args.rb)

if __name__ == "__main__":
    main()