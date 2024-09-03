

import numpy as np
import os

def load_data_from_file(filename):
    """
    Load the data from a text file and return it as a dictionary.
    """
    sets_data = {}
    current_set = None

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Set"):
                # New set ID
                current_set = line.split()[1].rstrip(":")
                sets_data[current_set] = {}  # Initialize a new dictionary for this set
            elif line.startswith("Part"):
                if current_set is None:
                    raise ValueError("Part found before set ID in file.")
                # Part ID and quantity
                parts = line.split(':')
                part_id = parts[0].split()[1]
                quantity = int(parts[1].strip())
                sets_data[current_set][part_id] = quantity

    return sets_data

def create_matrix(sets_data):
    """
    Create a matrix from the given sets data using numpy.
    """
    # Get unique set IDs and part IDs
    set_ids = list(sets_data.keys())
    part_ids = sorted(set(part_id for parts in sets_data.values() for part_id in parts))

    # Initialize a numpy matrix of zeros
    matrix = np.zeros((len(set_ids), len(part_ids)), dtype=int)

    # Fill the matrix with quantities
    for i, set_id in enumerate(set_ids):
        for part_id, quantity in sets_data[set_id].items():
            j = part_ids.index(part_id)
            matrix[i, j] = quantity

    return matrix, set_ids, part_ids

def write_output_to_file(sets_data, filename):
    """
    Write the sets data to a text file in the format:
    set_id: part_id: quantity
    """
    with open(filename, 'w') as file:
        for set_id, parts in sets_data.items():
            for part_id, quantity in parts.items():
                file.write(f"{set_id}: {part_id}: {quantity}\n")

def main():
    print("hello we are abotu to print the working directory")
    # Check the current working directory
    print("Current Working Directory:", os.getcwd())

    # Load data from file
    filename = 'sets.txt'  # Ensure this is the correct filename and path
    sets_data = load_data_from_file(filename)
    
    # Create the matrix using numpy
    matrix, set_ids, part_ids = create_matrix(sets_data)

    # Display the matrix
    print("Matrix:")
    print(matrix)

    # Display headers for better understanding
    print("\nSet IDs:", set_ids)
    print
if __name__ == "__main__":
    main()