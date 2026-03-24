import numpy as np

def fuzzy_c_means(data, n_clusters, m=2, error=0.005, max_iter=100, init_centers=None):
    """Implementación de Bezdek 1981."""
    n_points, n_features = data.shape
    
    # Inicializar centros (si no vienen del subtractive)
    if init_centers is not None and len(init_centers) >= n_clusters:
        centers = init_centers[:n_clusters]
    else:
        indices = np.random.choice(n_points, n_clusters, replace=False)
        centers = data[indices]
    
    # Matriz de pertenencia aleatoria (sume 1 por fila)
    u = np.random.dirichlet(np.ones(n_clusters), size=n_points).T 

    for _ in range(max_iter):
        u_prev = u.copy()
        
        # Actualizar centros
        um = u ** m
        centers = (um @ data) / np.sum(um, axis=1, keepdims=True)
        
        # Actualizar matriz U
        for i in range(n_points):
            dist = np.linalg.norm(data[i] - centers, axis=1)
            dist = np.where(dist == 0, 1e-10, dist)
            u[:, i] = 1.0 / np.sum((dist[:, None] / dist)**(2/(m-1)), axis=0)
            
        if np.linalg.norm(u - u_prev) < error:
            break
            
    return centers, u