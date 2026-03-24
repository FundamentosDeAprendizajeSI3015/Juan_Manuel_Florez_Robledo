import numpy as np

def subtractive_clustering(data, ra=0.5, rb=1.5, accept_ratio=0.5, reject_ratio=0.15):
    """Implementación de Chiu 1994."""
    n_points = data.shape[0]
    potential = np.zeros(n_points)
    
    # Paso 1: Calcular potencial inicial para cada punto
    for i in range(n_points):
        potential[i] = np.sum(np.exp(-4 * np.sum((data[i] - data)**2, axis=1) / (ra**2)))
    
    centers = []
    max_pot = np.max(potential)
    p_star = max_pot
    idx = np.argmax(potential)
    x_star = data[idx]
    
    # Paso 2: Iterar para encontrar centros
    while p_star > reject_ratio * max_pot:
        is_center = False
        if p_star > accept_ratio * max_pot:
            is_center = True
        else:
            d_min = np.min([np.linalg.norm(x_star - c) for c in centers]) if centers else 0
            if (d_min / ra) + (p_star / max_pot) >= 1:
                is_center = True
        
        if is_center:
            centers.append(x_star)
            # Paso 3: Reducir el potencial de los puntos cercanos al nuevo centro
            potential -= p_star * np.exp(-4 * np.sum((x_star - data)**2, axis=1) / (rb**2))
            potential[potential < 0] = 0
        else:
            potential[idx] = 0 # Eliminar este punto como candidato
            
        p_star = np.max(potential)
        idx = np.argmax(potential)
        x_star = data[idx]
            
    return np.array(centers)