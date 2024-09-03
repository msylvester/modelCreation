import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_model(model_filename='svd_trained_model.pkl'):
    """
    Load the trained SVD model from a pickle file.
    
    :param model_filename: The name of the pickle file containing the trained model.
    :return: The loaded SVD model.
    """
    return joblib.load(model_filename)

def read_csv_header(csv_filename='output.csv'):
    """
    Read the header from the CSV file to get part IDs.
    
    :param csv_filename: The name of the CSV file containing the matrix data.
    :return: List of part IDs.
    """
    df = pd.read_csv(csv_filename, nrows=0)
    return df.columns.tolist()[1:]  # Exclude the 'set_id' column

def prepare_query_vector(part_quantities, part_ids):
    """
    Prepare the query vector from given part quantities.
    
    :param part_quantities: Dictionary of part quantities with part IDs as keys.
    :param part_ids: List of part IDs from the header.
    :return: The query vector as a NumPy array.
    """
    # Initialize quantities with 0
    query_vector = np.zeros(len(part_ids))
    
    for part_id, quantity in part_quantities.items():
        if part_id in part_ids:
            query_vector[part_ids.index(part_id)] = quantity
    
    return query_vector

def predict_set(query_vector, svd_model, scaler, matrix):
    """
    Predict the most likely set based on the query vector and the trained model.
    
    :param query_vector: The query vector with part quantities.
    :param svd_model: The trained SVD model.
    :param scaler: The scaler used to normalize the training data.
    :param matrix: The original training matrix.
    :return: The index of the predicted set.
    """
    # Normalize the query vector
    query_vector_normalized = scaler.transform([query_vector])
    
    # Reduce dimensionality of the query vector
    query_vector_reduced = svd_model.transform(query_vector_normalized)
    
    # Reduce dimensionality of the training matrix
    matrix_reduced = svd_model.transform(matrix)
    
    # Find the closest set by computing the distance between the query vector and all sets
    distances = np.linalg.norm(matrix_reduced - query_vector_reduced, axis=1)
    closest_set_index = np.argmin(distances)
    
    return closest_set_index

def main():
    # Load the trained model and scaler
    svd_model = load_model('svd_trained_model.pkl')
    
    # Load the matrix and scaler used during training
    matrix = np.load('processed_matrix.npy')
    scaler = StandardScaler().fit(matrix)
    
    # Read part IDs from the CSV file
    part_ids = read_csv_header('output.csv')
    
    # Example input: Dictionary of part quantities
    part_quantities = {
       '6141': 2
        # '3068bpr9865': 8,
        # '52072': 6,
        # '71623': 3
        # Add more parts as needed
    }
    
    # Prepare the query vector
    query_vector = prepare_query_vector(part_quantities, part_ids)
    
    # Predict the set
    predicted_set_index = predict_set(query_vector, svd_model, scaler, matrix)
    
    print(f"Predicted set index: {predicted_set_index}")

if __name__ == "__main__":
    main()
