import os
import json

def update_catalog(catalog_path, catalog_images_dir):
    # Read the original catalog
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
    
    # Filter out items whose images were not downloaded
    updated_catalog = []
    for item in catalog:
        image_path = item['image_path']
        if os.path.exists(image_path):
            updated_catalog.append(item)
    
    # Save the updated catalog
    with open(catalog_path, 'w') as f:
        json.dump(updated_catalog, f, indent=4)
    
    print(f"Catalog updated. Original items: {len(catalog)}, Updated items: {len(updated_catalog)}")

if __name__ == "__main__":
    catalog_path = "data/catalog.json"
    catalog_images_dir = "data/catalog_images"
    update_catalog(catalog_path, catalog_images_dir) 