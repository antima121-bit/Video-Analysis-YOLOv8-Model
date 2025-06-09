import os
import json

def generate_final_output(matches_dir, vibes_dir, output_path):
    """
    Generate final output in the required format:
    {
        "video_id": "abc123",
        "vibes": ["Coquette", "Evening"],
        "products": [
            {
                "type": "dress",
                "color": "black",
                "match_type": "similar",
                "matched_product_id": "prod_456",
                "confidence": 0.84
            }
        ]
    }
    """
    final_output = []
    
    # Process each video's results
    for video_dir in os.listdir(matches_dir):
        if video_dir.endswith('_matches.json'):
            video_id = video_dir.replace('_matches.json', '')
            
            # Load matches
            with open(os.path.join(matches_dir, video_dir), 'r') as f:
                matches = json.load(f)
            
            # Load vibes
            vibe_file = os.path.join(vibes_dir, f"{video_id}_vibe.json")
            if os.path.exists(vibe_file):
                with open(vibe_file, 'r') as f:
                    vibe_data = json.load(f)
                    # Extract vibes list from the data
                    vibes = vibe_data.get('vibes', [])
                    if isinstance(vibes, list) and vibes and isinstance(vibes[0], dict):
                        vibes = [v['vibe'] for v in vibes]
            else:
                vibes = ["Casual"]  # Default vibe if not found
            
            # Format products (limit to top 30)
            formatted_products = []
            for product in matches[:30]:  # Take top 30 products
                formatted_product = {
                    "type": product.get("type", "unknown"),
                    "color": product.get("color", "unknown"),
                    "match_type": product.get("match_type", "similar"),
                    "matched_product_id": product.get("matched_product_id", "unknown"),
                    "confidence": float(product.get("confidence", 0.0))
                }
                formatted_products.append(formatted_product)
            
            # Create video result
            video_result = {
                "video_id": video_id,
                "vibes": vibes,
                "products": formatted_products
            }
            
            final_output.append(video_result)
    
    # Save final output
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=4)
    
    print(f"Final output generated and saved to {output_path}")
    print(f"Processed {len(final_output)} videos")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate final output JSON with video ID, vibes, and matched products")
    parser.add_argument("--matches_dir", type=str, default="data/matches", help="Directory containing match results")
    parser.add_argument("--vibes_dir", type=str, default="data/vibes", help="Directory containing vibe classification results")
    parser.add_argument("--output_path", type=str, default="data/final_output.json", help="Path to save the final output JSON")
    args = parser.parse_args()
    generate_final_output(args.matches_dir, args.vibes_dir, args.output_path) 