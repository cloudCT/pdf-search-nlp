# Script to fetch metadata from arXiv using arxivscraper

import pandas as pd
from arxivscraper import Scraper

from arxivscraper import Scraper
import pandas as pd
from .save_metadata import save_metadata_to_csv

def fetch_metadata(start_date="2025-04-01", end_date="2025-04-30", category="cs.LG"):
    """
    Fetch arXiv metadata for a given category (default 'cs.LG') and date range.
    Returns (metadata_df, id_list):
        metadata_df: pandas DataFrame of all records
        id_list: list of arXiv IDs (str)
    """
    main_category = category.split('.')[0]
    metadata_list = Scraper(
        category=main_category,
        date_from=start_date,
        date_until=end_date,
        filters={"categories": [category]}
    ).scrape()
    metadata_df = pd.DataFrame(metadata_list)
    id_list = [meta.get("id") for meta in metadata_list if meta.get("id")]
    return metadata_df, id_list

if __name__ == "__main__":
    metadata_df, id_list = fetch_metadata()
    print(f"Fetched {len(metadata_df)} records.")
    print(f"Fetched {len(id_list)} ids.")
    save_metadata_to_csv(metadata_df)

