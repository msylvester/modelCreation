"""
Create a matrix 
P
P where rows represent different sets and columns represent different pieces.
"""

import numpy as np

# Step 1: Load the data from files
with open('sets_id.txt', 'r') as f:
    set_ids = f.read().splitlines()  # List of set IDs

with open('part_ids.txt', 'r') as f:
    part_ids = f.read().splitlines()  # List of part IDs

# Step 2: Create the feature matrix P
# Assume we have a function get_piece_count(set_id, part_id) that returns the count of part_id in set_id

# Initialize the feature matrix P
P = np.zeros((len(set_ids), len(part_ids)))

# # Populate the matrix
# for i, set_id in enumerate(set_ids):
#     for j, part_id in enumerate(part_ids):
#         # Get the count of part_id in set_id
#         piece_count = get_piece_count(set_id, part_id)  # This function needs to be defined based on your data source
#         P[i, j] = piece_count

# # Example: P is now filled with the counts of each piece type in each set

[ 0 0 1,  0 2 0.  ]