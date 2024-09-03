def combine_files(file_list, output_filename):
    """
    Combine the contents of multiple files into a single output file.
    
    Parameters:
    - file_list: List of file paths to be combined.
    - output_filename: The path of the output file where the combined content will be written.
    """
    with open(output_filename, 'w') as outfile:
        for file_path in file_list:
            try:
                with open(file_path, 'r') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write('\n')  # Add a newline between contents of different files
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except Exception as e:
                print(f"An error occurred while reading {file_path}: {e}")

def main():
    # List of input files to be combined
    file_list = [
        '/Users/mess323/pytthonLeggo/part-quanity-in-set-eight.txt',
        '/Users/mess323/pytthonLeggo/part-quanity-in-set-eleven.txt',
        '/Users/mess323/pytthonLeggo/part-quanity-in-set-five.txt',
        '/Users/mess323/pytthonLeggo/part-quanity-in-set-four.txt',
        '/Users/mess323/pytthonLeggo/part-quanity-in-set-nine.txt',
        '/Users/mess323/pytthonLeggo/part-quanity-in-set-one.txt',
        '/Users/mess323/pytthonLeggo/part-quanity-in-set-seveb.txt',
        '/Users/mess323/pytthonLeggo/part-quanity-in-set-ten.txt',
        '/Users/mess323/pytthonLeggo/part-quanity-in-set-three.txt',
        '/Users/mess323/pytthonLeggo/part-quanity-in-set-two.txt',
        '/Users/mess323/pytthonLeggo/part-quanity-in-set.txt'
    ]
    
    # Output file where the combined contents will be written
    output_filename = '/Users/mess323/pytthonLeggo/combined_output.txt'
    
    combine_files(file_list, output_filename)
    
    print(f"Contents of files have been combined into {output_filename}.")

if __name__ == "__main__":
    main()
