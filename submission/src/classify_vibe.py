import os
import json
from typing import List, Dict, Any
from collections import Counter

class VibeClassifier:
    def __init__(self, vibes_path: str):
        """
        Initialize vibe classifier
        
        Args:
            vibes_path: Path to vibes list JSON file
        """
        # Load vibes
        with open(vibes_path, 'r') as f:
            self.vibes = json.load(f)
        
        # Define vibe rules
        self.vibe_rules = {
            "Coquette": ["pink", "lace", "floral", "romantic", "feminine", "delicate"],
            "Clean Girl": ["white", "minimal", "neutral", "simple", "elegant", "basic"],
            "Cottagecore": ["floral", "vintage", "natural", "rustic", "romantic", "pastel"],
            "Streetcore": ["urban", "edgy", "bold", "casual", "sporty", "modern"],
            "Y2K": ["retro", "vibrant", "playful", "colorful", "fun", "nostalgic"],
            "Boho": ["ethnic", "flowy", "natural", "hippie", "artistic", "free-spirited"],
            "Party Glam": ["sparkle", "bold", "elegant", "glamorous", "dressy", "luxurious"]
        }
    
    def _get_vibe_score(self, product: Dict[str, Any], vibe: str) -> float:
        """
        Get vibe score for product
        
        Args:
            product: Product information
            vibe: Vibe to check
        
        Returns:
            Vibe score
        """
        score = 0.0
        keywords = self.vibe_rules[vibe]
        
        # Check product type
        product_type = product['type'].lower()
        if any(keyword in product_type for keyword in keywords):
            score += 0.5
        
        # Check color
        color = product['color'].lower()
        if any(keyword in color for keyword in keywords):
            score += 0.5
        
        return score
    
    def classify_matches(self, matches: List[Dict[str, Any]]) -> List[str]:
        """
        Classify vibes based on matched products
        
        Args:
            matches: List of matched products
        
        Returns:
            List of top vibes
        """
        # Calculate vibe scores
        vibe_scores = {vibe: 0.0 for vibe in self.vibes}
        
        for match in matches:
            for product in match['matches']:
                for vibe in self.vibes:
                    score = self._get_vibe_score(product, vibe)
                    vibe_scores[vibe] += score * product['confidence']
        
        # Get top vibes
        sorted_vibes = sorted(vibe_scores.items(), key=lambda x: x[1], reverse=True)
        return [vibe for vibe, score in sorted_vibes[:3] if score > 0]
    
    def process_matches(self, matches_path: str, output_dir: str) -> str:
        """
        Process matches and classify vibes
        
        Args:
            matches_path: Path to matches JSON file
            output_dir: Directory to save classifications
        
        Returns:
            Path to classifications file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load matches
        with open(matches_path, 'r') as f:
            matches = json.load(f)
        
        # Classify vibes
        vibes = self.classify_matches(matches)
        
        # Save classifications
        output_path = os.path.join(output_dir, 'vibes.json')
        with open(output_path, 'w') as f:
            json.dump({
                'matches_file': matches_path,
                'vibes': vibes
            }, f, indent=2)
        
        return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify vibes from matches")
    parser.add_argument("--matches", required=True, help="Path to matches JSON file")
    parser.add_argument("--vibes", required=True, help="Path to vibes list JSON file")
    parser.add_argument("--output", required=True, help="Output directory for classifications")
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = VibeClassifier(args.vibes)
    
    # Process matches
    output_path = classifier.process_matches(args.matches, args.output)
    print(f"Saved vibe classifications to {output_path}") 