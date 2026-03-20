import numpy as np
from sklearn.decomposition import PCA

def hyperspectral_pca_reduction(data_cube, variance_threshold=0.95):
    # Hyperspectral Image Dimensionality Reduction using PCA
    H, W, B = data_cube.shape
    
    # Flatten the spatial dimensions to shape (H*W, B) for PCA
    flattened_data = data_cube.reshape((H * W, B))
    
    # Initialize PCA to retain >=95% of the information
    pca = PCA(n_components=variance_threshold)
    reduced_flattened = pca.fit_transform(flattened_data)
    
    # The optimal number of components K (stable between 8 and 12 for olivine data)
    K = pca.n_components_
    print(f"Optimal number of bands retained: {K}")
    
    # Reshape back to the 3D spatial cube (H, W, K)
    reduced_cube = reduced_flattened.reshape((H, W, K))
    return reduced_cube, pca
