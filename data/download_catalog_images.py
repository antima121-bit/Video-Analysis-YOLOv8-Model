import json
import requests
import os
from tqdm import tqdm
from pathlib import Path

def download_catalog_images(catalog_json_path):
    # Create directory for catalog images
    os.makedirs("data/catalog_images", exist_ok=True)
    
    # Load catalog
    with open(catalog_json_path, 'r') as f:
        catalog = json.load(f)
    
    # Download images
    for item in tqdm(catalog, desc="Downloading catalog images"):
        image_path = item['image_path']
        if not os.path.exists(image_path):
            try:
                response = requests.get(item['image_url'])
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                else:
                    print(f"Failed to download image for product {item['id']}")
            except Exception as e:
                print(f"Error downloading image for product {item['id']}: {str(e)}")

if __name__ == "__main__":
    catalog_json_path = "data/catalog.json"
    download_catalog_images(catalog_json_path) 