import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
print("About to read")
df = pd.read_csv('output.csv', index_col=0)  # index_col=0 assumes the first column is the index
print("Finsished reading about to create matrix ")
# Convert the DataFrame to a NumPy array

matrix = df.to_numpy()
# Save the NumPy array to a .npy file
np.save('processed_matrix.npy', matrix)
'''
1. right now we are verifying that we can get the matrix ready
2  then we will run the matrix similartity algo on a prediction vector
3. create a prediction vector
'''