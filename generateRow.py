import pandas as pd
import numpy as np

def generate_random_row(part_ids, set_id="new_set"):
    """
    Generate a new row with random quantities for each part ID.
    
    :param part_ids: List of part IDs.
    :param set_id: The set ID for the new row.
    :return: Dictionary representing the new row with set ID and random quantities.
    """
    random_quantities = {part_id: np.random.randint(0, 11) for part_id in part_ids}  # Random quantities between 0 and 10
    random_quantities['set_id'] = set_id
    return random_quantities

def main():
    # Load the existing CSV file into a DataFrame
    print('about to read')
    df = pd.read_csv('output.csv')
    
    # Get the list of part IDs (all columns except 'set_id')
    print('abotu to do some columns hit')
    part_ids = df.columns.tolist()
    part_ids.remove('set_id')

    # Generate a new row with random quantities
    print('abotu to do some some random shit')
    new_row = generate_random_row(part_ids)
    
    # Append the new row to the DataFrame
    print('about to append some shit')
    df = df.append(new_row, ignore_index=True)

    # Save the updated DataFrame back to the CSV file or a new CSV file
    df.to_csv('output_with_random_row.csv', index=False)
    print("New row with random quantities added to 'output_with_random_row.csv'.")

if __name__ == "__main__":
    main()
