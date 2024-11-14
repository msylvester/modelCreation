# 🧱 LEGO Set Identifier with K-Nearest Neighbors 🧩
This project uses a K-Nearest Neighbors (KNN) model to identify potential LEGO set IDs based on a list of LEGO parts you provide. With a trained model and a parts list, you can find matching LEGO sets from your collection! Perfect for enthusiasts, collectors, and developers. 🎉

## ✨ Features

- 🔍 LEGO Set Matching: Input a dictionary of parts and quantities, and the KNN model will return potential matching set IDs.
- 🧠 Cosine Similarity Matching: Utilizes cosine similarity to find the closest matches.
- 💾 Save and Load Models: Trained models and scalers are saved for easy re-use.

## 🚀 Getting Started
📋 Prerequisites
- Python 3.7+
- pandas, scikit-learn, numpy, joblib (Install with pip install -r requirements.txt)

## 📥 Installation


1. Install dependencies:

```bash

pip install -r requirements.txt
```

🛠️ Usage
Prepare the Data:

Ensure dataframe.pkl exists, containing LEGO set data with part quantities.
Run the Script:

Train the model and predict sets for a parts list:
bash
Copy code
python script_name.py
Make Predictions:

Example dictionary format for parts:
python
Copy code
parts_list = {
    '6141': 2,
    '3001': 4,
    '3003': 1
}
Expected Output:

Closest set IDs for the given parts list:
json
Copy code
{
  "set_ids": ["1234-1", "5678-1"]
}
📤 Example Request
Using curl:

```bash

curl -X POST http://localhost:5000/identify -H "Content-Type: application/json" -d '{"parts": {"6141": 2, "3001": 4}}'
```

📂 Files in Project
- dataframe.pkl: Pickle file with LEGO set data.
- knn_model.pkl: Saved KNN model file.
- scaler.pkl: Saved scaler for normalizing parts quantities.
📜 License
This project is open-source. Feel free to use, modify, and share! 🎉

