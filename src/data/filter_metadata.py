import os
import pandas as pd
from src.utils.config import DATA_PATH

def filter_metadata(metadata_csv=None, raw_dir=None, processed_dir=None, output_csv=None):
    """
    Filter metadata to only include entries with both a raw PDF and processed TXT file present.
    Args:
        metadata_csv: Path to the metadata CSV file.
        raw_dir: Directory containing raw PDFs.
        processed_dir: Directory containing processed TXTs.
        output_csv: Path to save the filtered metadata CSV.
    Returns:
        filtered_df: The filtered pandas DataFrame.
    """
    if metadata_csv is None:
        metadata_csv = os.path.join(DATA_PATH, 'metadata.csv')
    if raw_dir is None:
        raw_dir = os.path.join(DATA_PATH, 'raw')
    if processed_dir is None:
        processed_dir = os.path.join(DATA_PATH, 'processed')
    if output_csv is None:
        output_csv = os.path.join(DATA_PATH, 'metadata.csv')

    metadata_df = pd.read_csv(metadata_csv)
    # Extract arXiv IDs from filenames
    raw_ids = set(os.path.splitext(f)[0] for f in os.listdir(raw_dir) if f.endswith('.pdf'))
    processed_ids = set(os.path.splitext(f)[0] for f in os.listdir(processed_dir) if f.endswith('.txt'))
    # Only keep entries where both files exist
    valid_ids = raw_ids & processed_ids
    filtered_df = metadata_df[metadata_df['id'].astype(str).isin(valid_ids)].copy()
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered metadata: {len(filtered_df)} entries saved to {output_csv} (from {len(metadata_df)} original entries)")
    return filtered_df

if __name__ == "__main__":
    filter_metadata()
