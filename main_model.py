import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

def load_dataframe_from_pickle(pickle_filename='dataframe.pkl'):
    df = pd.read_pickle(pickle_filename)
    df = df.astype(pd.SparseDtype("float", fill_value=0))  # Use sparse data
    return df

def prepare_data_for_knn(df):
    X = df.values  # Features: quantities of parts
    y = df.index.values  # Set IDs

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction (optional)
    X_reduced, pca = reduce_dimensions(X_scaled)

    return X_reduced, y, scaler, pca

def reduce_dimensions(X_scaled, n_components=100):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced, pca

def train_knn_model(X_scaled, n_neighbors=5):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(X_scaled)
    return knn

def save_model(model, scaler, pca, model_filename='knn_model.pkl', scaler_filename='scaler.pkl', pca_filename='pca.pkl'):
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(pca, pca_filename)
    print(f"Model saved to '{model_filename}', scaler to '{scaler_filename}', PCA to '{pca_filename}'.")

def predict_sets_for_parts(knn_model, scaler, pca, parts_list, df, y, n_results=5):
    part_vector = np.zeros((1, df.shape[1]))  # Vector size is the number of columns (parts)
    for part, quantity in parts_list.items():
        if part in df.columns:
            index = df.columns.get_loc(part)
            part_vector[0, index] = quantity

    # Normalize and reduce dimensionality of the part vector
    part_vector_scaled = scaler.transform(part_vector)
    part_vector_reduced = pca.transform(part_vector_scaled)

    # Find nearest neighbors
    distances, indices = knn_model.kneighbors(part_vector_reduced, n_neighbors=n_results)
    closest_set_ids = y[indices.flatten()]
    return closest_set_ids

def main():
    df = load_dataframe_from_pickle('dataframe.pkl')
    X_scaled, y, scaler, pca = prepare_data_for_knn(df)
    knn_model = train_knn_model(X_scaled, n_neighbors=5)
    save_model(knn_model, scaler, pca)

    parts_list = {'6141': 2}  # Example input

    closest_set_ids = predict_sets_for_parts(knn_model, scaler, pca, parts_list, df, y)
    print(f"Closest set IDs: {closest_set_ids}")

if __name__ == "__main__":
    main()
