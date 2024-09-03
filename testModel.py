import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
import numpy as np

def load_dataframe_from_pickle(pickle_filename='dataframe.pkl'):
    """
    Load the pandas DataFrame from a pickle file.
    
    :param pickle_filename: The name of the pickle file containing the DataFrame.
    :return: The loaded DataFrame.
    """
    df = pd.read_pickle(pickle_filename)
    return df
def load_model_and_scaler(model_filename='knn_model.pkl', scaler_filename='scaler.pkl'):
    """
    Load the KNN model and scaler from files.
    
    :param model_filename: The filename of the saved model.
    :param scaler_filename: The filename of the saved scaler.
    :return: The loaded KNN model and scaler.
    """
    knn_model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    print(f"Model loaded from '{model_filename}' and scaler loaded from '{scaler_filename}'.")
    return knn_model, scaler


def predict_sets_for_parts(knn_model, scaler, parts_list, df, y, n_results=5):
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

    # Example: Predict sets for a given list of parts
    parts_list = {
        '6141': 2
        
        # Add more parts and their quantities as needed
    }
        # Load the DataFrame from the pickle file
    df = load_dataframe_from_pickle('dataframe.pkl')
    y = df.index.values  # Set IDs

    knn_model, scaler = load_model_and_scaler('knn_model.pkl', 'scaler.pkl')

    closest_set_ids = predict_sets_for_parts(knn_model, scaler, parts_list, df, y)
    print(f"Closest set IDs for the given parts list: {closest_set_ids}")


if __name__ == "__main__":
    main()
