import os
import pandas as pd
import numpy as np

def split_csv_to_new_dir(input_dir, output_dir, n_splits=2):
    """
    Reads CSVs from input_dir and saves n split parts into output_dir
    while maintaining subfolder structure.
    """
    subfolders = ['reckless', 'normal']
    
    for folder in subfolders:
        # Define paths
        src_folder = os.path.join(input_dir, folder)
        dest_folder = os.path.join(output_dir, folder)
        
        # Create destination folder if it doesn't exist
        os.makedirs(dest_folder, exist_ok=True)
        
        if not os.path.exists(src_folder):
            print(f"Skipping: {src_folder} not found.")
            continue

        for file in os.listdir(src_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(src_folder, file)
                df = pd.read_csv(file_path)
                
                # Split the data
                df_splits = np.array_split(df, n_splits)
                base_name = os.path.splitext(file)[0]
                
                for i, part_df in enumerate(df_splits):
                    new_filename = f"{base_name}_part_{i+1}.csv"
                    save_path = os.path.join(dest_folder, new_filename)
                    
                    part_df.to_csv(save_path, index=False)
                    print(f"Exported: {save_path}")

# Configuration
INPUT_PATH = 'data/processed/2026-04-20'
OUTPUT_PATH = 'data/processed/2026-04-20_split'
N = 2 

if __name__ == "__main__":
    split_csv_to_new_dir(INPUT_PATH, OUTPUT_PATH, n_splits=N)
