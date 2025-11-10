import requests
import zipfile
import csv
import os
from pathlib import Path
from io import BytesIO
import time
import logging
import gzip


logger = logging.getLogger('GDELTScraper')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class GDELTScraper:
    def __init__(self, output_dir="media_data"):
        self.master_url = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        
    def download_master_list(self):
        """Download the master file list from GDELT"""
        logger.info("Downloading master file list...")
        response = requests.get(self.master_url)
        response.raise_for_status()
        return response.text.splitlines()
    
    def filter_gkg_urls(self, lines):
        """Filter lines to get only GKG file URLs"""
        gkg_urls = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                url = parts[2]
                if 'gkg' in url.lower():
                    gkg_urls.append(url)
        return gkg_urls
    
    def download_and_save_gkg(self, url, delay=1):
        """Download a GKG file, decompress it, and save as CSV"""
        filename = url.split('/')[-1]
        csv_filename = filename.replace('.csv.zip', '.csv').replace('.gkg.csv.zip', '.gkg.csv')
        output_path = self.output_dir / csv_filename
        
        # Skip if already downloaded
        if output_path.exists():
            logger.info(f"Skipping {filename} (already exists)")
            return True
        
        try:
            logger.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Decompress the gzipped content
            decompressed = gzip.decompress(response.content)
            csv_content = decompressed.decode('utf-8')
            
            # Save to disk
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                f.write(csv_content)
            
            logger.info(f"Saved: {csv_filename}")
            time.sleep(delay)  # Be respectful to the server
            return True
            
        except Exception as e:
            logger.info(f"Error downloading {filename}: {str(e)}")
            return False
    
    def run(self, max_files=None, delay=1):
        """Run the complete scraping process"""
        try:
            # Download and parse master list
            lines = self.download_master_list()
            logger.info(f"Master list downloaded: {len(lines)} entries")
            
            # Filter for GKG URLs
            gkg_urls = self.filter_gkg_urls(lines)
            logger.info(f"Found {len(gkg_urls)} GKG files")
            
            if max_files:
                gkg_urls = gkg_urls[:max_files]
                logger.info(f"Limiting to {max_files} files")
            
            # Download each GKG file
            successful = 0
            failed = 0
            
            for i, url in enumerate(gkg_urls, 1):
                logger.info(f"\nProcessing {i}/{len(gkg_urls)}")
                if self.download_and_save_gkg(url, delay):
                    successful += 1
                else:
                    failed += 1
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Download complete!")
            logger.info(f"Successful: {successful}")
            logger.info(f"Failed: {failed}")
            logger.info(f"Files saved to: {self.output_dir.absolute()}")
            
        except Exception as e:
            logger.info(f"Error in scraping process: {str(e)}")


if __name__ == "__main__":
    scraper = GDELTScraper(output_dir="gdelt_gkg_data")
    
    #test
    scraper.run(max_files=10, delay=1)
    
    # To download all files THIS WILL CRASH YOUR COMPUTER DO NOT RUN WITHOUT ADEQUATE RESOURCES
    # scraper.run(delay=1)
