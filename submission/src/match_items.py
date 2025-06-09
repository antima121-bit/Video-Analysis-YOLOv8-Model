import os
import json
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from typing import List, Dict, Any
from PIL import Image

class ItemMatcher:
    def __init__(self, catalog_index_path: str, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize item matcher
        
        Args:
            catalog_index_path: Path to catalog index JSON file
            model_name: Name of CLIP model to use
        """
        # Load catalog index
        with open(catalog_index_path, 'r') as f:
            self.catalog_index = json.load(f)
        
        # Initialize CLIP
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Convert catalog features to tensor
        self.catalog_features = torch.tensor(
            [item['features'] for item in self.catalog_index]
        ).to(self.device)
    
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
        return features
    
    def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """
        Compute cosine similarity between feature vectors
        
        Args:
            features1: First feature vector
            features2: Second feature vector
        
        Returns:
            Similarity score
        """
        similarity = torch.nn.functional.cosine_similarity(features1, features2)
        return float(similarity)
    
    def find_matches(self, image: Image.Image, threshold: float = 0.75, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find matching products for image
        
        Args:
            image: Input image
            threshold: Similarity threshold
            top_k: Number of top matches to return
        
        Returns:
            List of matched products
        """
        # Get image features
        features = self.get_image_features(image)
        
        # Compute similarities
        similarities = torch.nn.functional.cosine_similarity(
            features, self.catalog_features
        )
        
        # Get top matches
        top_indices = torch.topk(similarities, top_k).indices
        matches = []
        
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity >= threshold:
                product = self.catalog_index[idx]
                matches.append({
                    'type': product['category'],
                    'color': product['color'],
                    'matched_product_id': product['product_id'],
                    'match_type': 'exact' if similarity > 0.9 else 'similar',
                    'confidence': similarity
                })
        
        return matches
    
    def process_items(self, items_dir: str, output_dir: str, threshold: float = 0.75) -> str:
        """
        Process all items in directory
        
        Args:
            items_dir: Directory containing cropped items
            output_dir: Directory to save matches
            threshold: Similarity threshold
        
        Returns:
            Path to matches file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each item
        matches = []
        for item_file in tqdm(os.listdir(items_dir), desc="Processing items"):
            if not item_file.endswith('.jpg'):
                continue
            
            # Load image
            image = Image.open(os.path.join(items_dir, item_file))
            
            # Find matches
            item_matches = self.find_matches(image, threshold)
            
            if item_matches:
                matches.append({
                    'item': item_file,
                    'matches': item_matches
                })
        
        # Save matches
        output_path = os.path.join(output_dir, 'matches.json')
        with open(output_path, 'w') as f:
            json.dump(matches, f, indent=2)
        
        return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Match items with catalog")
    parser.add_argument("--catalog-index", required=True, help="Path to catalog index JSON file")
    parser.add_argument("--items", required=True, help="Directory containing cropped items")
    parser.add_argument("--output", required=True, help="Output directory for matches")
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="CLIP model name")
    
    args = parser.parse_args()
    
    # Initialize matcher
    matcher = ItemMatcher(args.catalog_index, args.model)
    
    # Process items
    output_path = matcher.process_items(args.items, args.output, args.threshold)
    print(f"Saved matches to {output_path}") 