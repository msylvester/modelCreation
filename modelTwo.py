import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib

def load_dataframe_from_pickle(pickle_filename='dataframe.pkl'):
    """
    Load the pandas DataFrame from a pickle file.
    
    :param pickle_filename: The name of the pickle file containing the DataFrame.
    :return: The loaded DataFrame.
    """
    df = pd.read_pickle(pickle_filename)
    return df

def prepare_data_for_model(df):
    """
    Prepare the data for supervised model training with part IDs included.
    
    :param df: The pandas DataFrame containing the matrix data.
    :return: The features (X) and target (y) arrays.
    """
    print("DataFrame columns:", df.columns)
    print("DataFrame index:", df.index)

    # In this case, the DataFrame index is the set IDs
    set_ids = df.index.values  # Target: set IDs
    
    # Features: quantities of parts
    X = df.values  
    y = set_ids
    
    # Encode the set IDs to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature Selection
    feature_selector = RandomForestClassifier(n_estimators=10, random_state=42)
    feature_selector.fit(X_scaled, y_encoded)
    
    selector = SelectFromModel(feature_selector, threshold='mean', prefit=True)
    X_selected = selector.transform(X_scaled)
    
    return X_selected, y_encoded, label_encoder

def train_model(X_train, y_train):
    """
    Train a supervised model on the training data.
    
    :param X_train: The training features.
    :param y_train: The training target.
    :return: The trained model.
    """
    model = Pipeline([
        ('svd', TruncatedSVD(n_components=50)),  # Example: 50 components
        ('clf', RandomForestClassifier(n_estimators=50, max_depth=10))  # Example classifier with reduced complexity
    ])
    
    model.fit(X_train, y_train)
    return model

def save_model(model, model_filename='svd_model.pkl'):
    """
    Save the trained model to a file.
    
    :param model: The model to be saved.
    :param model_filename: The filename of the model.
    """
    joblib.dump(model, model_filename, compress=3)  # Compress the model file to save space
    print(f"Model saved to '{model_filename}'.")

def main():
    # Load the DataFrame from the pickle file
    df = load_dataframe_from_pickle('dataframe.pkl')
    
    # Prepare data for the model
    X, y, label_encoder = prepare_data_for_model(df)
    
    # Split the data into training, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Validate the model (e.g., print validation score)
    val_score = model.score(X_val, y_val)
    print(f"Validation Score: {val_score:.2f}")
    
    # Save the trained model
    save_model(model, 'svd_model_two.pkl')
    
    print("Model training complete and saved.")

if __name__ == "__main__":
    main()
