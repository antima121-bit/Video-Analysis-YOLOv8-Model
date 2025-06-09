import os
import pandas as pd
import requests
from tqdm import tqdm
import json
import gdown

def download_file(url, output_path):
    """
    Download a file from Google Drive or direct URL
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if 'drive.google.com' in url:
        try:
            gdown.download(url, output_path, quiet=False)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Successfully downloaded to {output_path}")
            else:
                print(f"Warning: Downloaded file {output_path} is empty or does not exist")
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
    else:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc=os.path.basename(output_path),
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            print(f"Successfully downloaded to {output_path}")
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")

def download_videos():
    """
    Download all video files from the Google Drive folder
    """
    # Video file URLs (direct download links)
    video_files = {
        "2025-05-22_08-25-12_UTC.mp4": "https://drive.google.com/uc?export=download&id=1-Q_M3OnIANArnDzePapptVWEthzFDjL9",
        "2025-05-27_13-46-16_UTC.mp4": "https://drive.google.com/uc?export=download&id=1-Q_M3OnIANArnDzePapptVWEthzFDjL9",
        "2025-05-28_13-40-09_UTC.mp4": "https://drive.google.com/uc?export=download&id=1-Q_M3OnIANArnDzePapptVWEthzFDjL9",
        "2025-05-28_13-42-32_UTC.mp4": "https://drive.google.com/uc?export=download&id=1-Q_M3OnIANArnDzePapptVWEthzFDjL9",
        "2025-05-31_14-01-37_UTC.mp4": "https://drive.google.com/uc?export=download&id=1-Q_M3OnIANArnDzePapptVWEthzFDjL9",
        "2025-06-02_11-31-19_UTC.mp4": "https://drive.google.com/uc?export=download&id=1-Q_M3OnIANArnDzePapptVWEthzFDjL9"
    }
    
    for filename, url in video_files.items():
        output_path = os.path.join('data', 'videos', filename)
        print(f"\nDownloading {filename}...")
        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")

def setup_data():
    """
    Download and organize all data files
    """
    print("Creating data directories...")
    os.makedirs('data/videos', exist_ok=True)
    os.makedirs('data/catalog', exist_ok=True)
    os.makedirs('data/frames', exist_ok=True)
    os.makedirs('data/detections', exist_ok=True)
    
    print("\nDownloading vibes list...")
    vibes_url = "https://drive.google.com/uc?id=1-Q_M3OnIANArnDzePapptVWEthzFDjL9"
    download_file(vibes_url, 'data/vibes_list.json')
    
    print("\nDownloading product data...")
    product_data_url = "https://drive.google.com/uc?id=18y4M0FEzX8F1IifqjbnIwsLal0SZj0lo"
    download_file(product_data_url, 'data/catalog/product_data.xlsx')
    
    print("\nDownloading images.csv...")
    images_url = "https://drive.google.com/uc?id=1iyc0tGfCzKBqZUPQ7gHaD1DyjSsVCDUe"
    download_file(images_url, 'data/images.csv')
    
    print("\nStarting video downloads...")
    download_videos()
    
    print("\nConverting product data to CSV...")
    if os.path.exists('data/catalog/product_data.xlsx'):
        try:
            df = pd.read_excel('data/catalog/product_data.xlsx', engine='openpyxl')
            df.to_csv('data/catalog/product_data.csv', index=False)
            print("Successfully converted product data to CSV format")
        except Exception as e:
            print(f"Error converting Excel to CSV: {str(e)}")

if __name__ == "__main__":
    print("Starting data setup...")
    setup_data()
    print("\nData setup complete!") 