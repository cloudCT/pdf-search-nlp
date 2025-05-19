## Adding project root to sys.path for src imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


def extract_metadata(document):
    # Extract metadata fields from the document
    return {
        "id": document.get("id"),
        "title": document.get("title"),
        "abstract": document.get("abstract"),
        "categories": document.get("categories"),
        "doi": document.get("doi"),
        "created": document.get("created"),
        "updated": document.get("updated"),
        "authors": document.get("authors", []),
        "affiliation": document.get("affiliation"),
        "url": document.get("url"),
    }



