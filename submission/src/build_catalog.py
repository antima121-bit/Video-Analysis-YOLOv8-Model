import os
import json
import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import requests
from io import BytesIO

class CatalogIndexer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize catalog indexer
        
        Args:
            model_name: Name of CLIP model to use
        """
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def load_image(self, url: str) -> Image.Image:
        """
        Load image from URL
        
        Args:
            url: Image URL
        
        Returns:
            PIL Image
        """
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    
    def get_image_features(self, image: Image.Image) -> torch.Tensor:
        """
        Get image features using CLIP
        
        Args:
            image: Input image
        
        Returns:
            Image features tensor
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.cpu()
    
    def build_index(self, catalog_path: str, output_dir: str) -> str:
        """
        Build catalog index
        
        Args:
            catalog_path: Path to catalog CSV file
            output_dir: Directory to save index
        
        Returns:
            Path to index file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load catalog
        catalog = pd.read_csv(catalog_path)
        
        # Process each product
        index = []
        for _, product in tqdm(catalog.iterrows(), desc="Building index", total=len(catalog)):
            try:
                # Load image
                image = self.load_image(product['shopify_cdn_url'])
                
                # Get features
                features = self.get_image_features(image)
                
                # Add to index
                index.append({
                    'product_id': product['product_id'],
                    'title': product['title'],
                    'category': product['category'],
                    'color': product['color'],
                    'features': features.numpy().tolist()
                })
            except Exception as e:
                print(f"Error processing {product['product_id']}: {e}")
        
        # Save index
        output_path = os.path.join(output_dir, 'catalog_index.json')
        with open(output_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build catalog index")
    parser.add_argument("--catalog", required=True, help="Path to catalog CSV file")
    parser.add_argument("--output", required=True, help="Output directory for index")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="CLIP model name")
    
    args = parser.parse_args()
    
    # Initialize indexer
    indexer = CatalogIndexer(args.model)
    
    # Build index
    output_path = indexer.build_index(args.catalog, args.output)
    print(f"Saved catalog index to {output_path}") 