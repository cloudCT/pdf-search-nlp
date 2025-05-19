"""

Fetches arXiv metadata and PDFs for the cs.LG (Machine Learning) category using the arxivscraper Python library (https://github.com/lamphanviet/arxivscraper).

This script is a modern, robust alternative to OAI-PMH/Sickle-based harvesting. It directly queries the arXiv API for cs.LG papers in a given date range, saves metadata to CSV, and optionally downloads PDFs.
"""

import os
import sys
import pandas as pd
from arxivscraper import Scraper
import requests
from src.utils.config import DATA_PATH
from src.data.filter_metadata import filter_metadata

# Ensure the data directory exists
os.makedirs(DATA_PATH, exist_ok=True)



## Fetching metadata
# Returns list of dictionaries and list of id's

def fetch_metadata(start_date= "2025-04-01", end_date= "2025-04-30", category="cs.LG"):
    """
    Fetch arXiv metadata for a given category (default 'cs.LG') and date range.
    Returns (metadata_list, id_list).
    """
    # arxivscraper expects the top-level archive and filters by category
    main_category = category.split('.')[0]
    metadata_list = Scraper(
        category=main_category,
        date_from=start_date,
        date_until=end_date,
        filters={"categories": [category]}
    ).scrape()
    id_list = [meta.get("id") for meta in metadata_list if meta.get("id")]
    metadata_df = pd.DataFrame(metadata_list)
    return metadata_df, id_list


## Converting metadata to pd Dataframe and saving to csv

# artifact for now 
# def get_metadata_df(metadata_list):
#     """
#     Convert a list of metadata dictionaries to a pandas DataFrame.
#     """
#     metadata_df = pd.DataFrame(metadata_list)

#     return metadata_df

def save_metadata_to_csv(metadata_df):
    """
    Save DataFrame to CSV in the DATA_PATH directory.
    """
    path = os.path.join(DATA_PATH,"metadata.csv")
    metadata_df.to_csv(path, index=False)
    print(f"Saved metadata to {path}")


# Downloading pdf's

def download_pdfs(id_list):
    """
    Download PDFs for all arXiv IDs in id_list.
    Saves to DATA_PATH/raw. Tracks and saves failed IDs to DATA_PATH/failed_downloads.txt if any.
    Returns the list of failed IDs.
    """
    download_dir = os.path.join(DATA_PATH, "raw")
    os.makedirs(download_dir, exist_ok=True)
    total = len(id_list)
    success = 0
    fail = 0
    failed_ids = []
    for i, arxiv_id in enumerate(id_list):
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_path = os.path.join(download_dir, f"{arxiv_id}.pdf")
        try:
            response = requests.get(pdf_url)
            if response.status_code == 200:
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
                success += 1
                if success % 100 == 0:
                    print(f"Downloaded {success} PDFs so far (latest: {arxiv_id})")
            else:
                print(f"Failed to download {arxiv_id}: HTTP {response.status_code}")
                fail += 1
                failed_ids.append(arxiv_id)
        except Exception as ex:
            print(f"Exception downloading {arxiv_id}: {ex}")
            fail += 1
            failed_ids.append(arxiv_id)
    print(f"Download complete. Success: {success}, Failed: {fail}, Total attempted: {total}")
    # Save failed IDs to file if there were any failures
    if failed_ids:
        failed_path = os.path.join(DATA_PATH, "failed_downloads.txt")
        with open(failed_path, "w") as f:
            for fid in failed_ids:
                f.write(fid + "\n")
        print(f"Saved {len(failed_ids)} failed IDs to {failed_path}")
    return failed_ids


###### Test run:
# Note: This is a test run and will only fetch a sample of 500 records.

metadata_df, id_list = fetch_metadata()
print(f"Fetched {len(metadata_df)} records.")
print(f"Fetched {len(id_list)} ids.")

metadata_sample = metadata_df.sample(n=500, random_state=42)
sample_ids = metadata_sample['id'].tolist()

save_metadata_to_csv(metadata_sample)
failed_ids = download_pdfs(sample_ids)
filter_metadata()
print(f"Failed to download {len(failed_ids)} ids.")




#####

if __name__ == "__main__":
    # Set date range for cs.LG papers
    start_date = "2025-04-01"
    end_date = "2025-04-30"
    # Fetch metadata and IDs
    metadata_df, id_list = fetch_metadata(start_date, end_date)
    print(f"Fetched {len(metadata_df)} records.")
    # Save metadata to CSV
    save_metadata_to_csv(metadata_df)
    # Download PDFs using the list of IDs
    download_pdfs(id_list)
    filter_metadata()
