import pandas as pd
import json
import os
from pathlib import Path

def convert_catalog_to_json(csv_path, output_path):
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Create catalog items list
    catalog_items = []
    for _, row in df.iterrows():
        item = {
            'id': str(row['id']),
            'image_url': str(row['image_url']),
            'image_path': f"data/catalog_images/{row['id']}.jpg"  # We'll download these images
        }
        catalog_items.append(item)
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(catalog_items, f, indent=4)
    
    print(f"Catalog converted and saved to {output_path}")
    return catalog_items

if __name__ == "__main__":
    csv_path = "data/images.csv"
    output_path = "data/catalog.json"
    convert_catalog_to_json(csv_path, output_path) 