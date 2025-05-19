# Script to process and extract text from PDFs using PyMuPDF

import os
import sys

# Robustly adding project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import fitz  # PyMuPDF
from src.utils.config import DATA_PATH
from src.data.filter_metadata import filter_metadata

# By default, use the standardized data/raw and data/processed directories from config

def process_pdfs(
    pdf_dir=os.path.join(DATA_PATH, 'raw'),
    output_dir=os.path.join(DATA_PATH, 'processed')
):
    """
    Extract text from all PDFs in pdf_dir using PyMuPDF and save as .txt files in output_dir.
    Handles errors gracefully and logs progress.
    Directories are set according to project config (DATA_PATH).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs to process in {pdf_dir}.")
    success = 0
    fail = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        output_file = os.path.join(output_dir, f'{os.path.splitext(pdf_file)[0]}.txt')
        try:
            with fitz.open(pdf_path) as doc:
                text = ''
                for page in doc:
                    text += page.get_text()
            # Clean control characters from extracted text
            def clean_text(text):
                import re
                return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
            text = clean_text(text)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            success += 1
            if success % 100 == 0:
                print(f"Extracted {success} PDFs so far (latest: {pdf_file})")
        except Exception as e:
            print(f"Failed to process {pdf_file}: {e}")
            fail += 1
    print(f"Done. Success: {success}, Failed: {fail}, Total: {len(pdf_files)}")

# Example usage
if __name__ == "__main__":
    process_pdfs()
    filter_metadata() # To filter metadata by existing pdf's and txt's
