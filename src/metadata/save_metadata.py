# Script to convert metadata to a pandas DataFrame and save as CSV

import pandas as pd

def save_metadata_to_csv(metadata_df, output_path="metadata_sample.csv"):
    """
    Save the metadata DataFrame to a CSV file.
    Args:
        metadata_df: pandas DataFrame
        output_path: path to save CSV (default: 'metadata_sample.csv')
    """
    metadata_df.to_csv(output_path, index=False)
    print(f"Saved metadata to {output_path}")

