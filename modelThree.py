import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib


def load_dataframe_from_pickle(pickle_filename='dataframe.pkl'):
    """
    Load the pandas DataFrame from a pickle file.
    
    :param pickle_filename: The name of the pickle file containing the DataFrame.
    :return: The loaded DataFrame.
    """
    df = pd.read_pickle(pickle_filename)
    return df


def prepare_data_for_knn(df):
    """
    Prepare the data for the K-Nearest Neighbors model.
    
    :param df: The pandas DataFrame containing the matrix data.
    :return: The features (X) array, the set IDs (y), and the part IDs (columns).
    """
    print("DataFrame columns (part IDs):", df.columns)
    print("DataFrame index (set IDs):", df.index)

    X = df.values  # Features: quantities of parts
    y = df.index.values  # Set IDs

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def train_knn_model(X_scaled, n_neighbors=5):
    """
    Train a K-Nearest Neighbors model on the data.
    
    :param X_scaled: The scaled features.
    :param n_neighbors: The number of neighbors to use for KNN.
    :return: The trained KNN model.
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(X_scaled)
    return knn


def save_model(model, scaler, model_filename='knn_model.pkl', scaler_filename='scaler.pkl'):
    """
    Save the trained KNN model and scaler to files.
    
    :param model: The model to be saved.
    :param scaler: The scaler used to normalize the features.
    :param model_filename: The filename of the model.
    :param scaler_filename: The filename of the scaler.
    """
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Model saved to '{model_filename}' and scaler saved to '{scaler_filename}'.")


def predict_sets_for_parts(knn_model, scaler, parts_list, df, n_results=5):
    """
    Predict the most similar sets for a given list of parts.
    
    :param knn_model: The trained KNN model.
    :param scaler: The scaler used to normalize the features.
    :param parts_list: A dictionary representing the parts and their quantities.
    :param df: The DataFrame containing the original data.
    :param y: The set IDs corresponding to each row in the DataFrame.
    :param n_results: The number of closest sets to return.
    :return: The closest set IDs.
    """
    # Convert the parts list to a vector with the same structure as the training data
    part_vector = np.zeros((1, df.shape[1]))  # Vector size is the number of columns (parts) in df
    for part, quantity in parts_list.items():
        if part in df.columns:
            index = df.columns.get_loc(part)
            part_vector[0, index] = quantity

    # Normalize the part vector
    part_vector_scaled = scaler.transform(part_vector)

    # Find the nearest neighbors
    distances, indices = knn_model.kneighbors(part_vector_scaled, n_neighbors=n_results)

    # Get the set IDs of the closest matches
    closest_set_ids = y[indices.flatten()]
    return closest_set_ids


def main():
    # Load the DataFrame from the pickle file
    df = load_dataframe_from_pickle('dataframe.pkl')

    # Prepare data for the KNN model
    X_scaled, y, scaler = prepare_data_for_knn(df)

    # Train the KNN model
    knn_model = train_knn_model(X_scaled, n_neighbors=5)

    # Save the trained model and scaler
    save_model(knn_model, scaler, 'knn_model.pkl', 'scaler.pkl')

    # Example: Predict sets for a given list of parts
    parts_list = {
        '6141': 2
        
        # Add more parts and their quantities as needed
    }

    closest_set_ids = predict_sets_for_parts(knn_model, scaler, parts_list, df)
    print(f"Closest set IDs for the given parts list: {closest_set_ids}")


if __name__ == "__main__":
    main()
