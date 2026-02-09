"""Download UCI dataset"""
import urllib.request
from pathlib import Path

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
output = Path("data/raw/Online_Retail.xlsx")

if output.exists():
    print(f"✅ File already exists: {output}")
else:
    print("📥 Downloading dataset (takes ~1 minute)...")
    urllib.request.urlretrieve(url, output)
    size_mb = output.stat().st_size / (1024*1024)
    print(f"✅ Downloaded! Size: {size_mb:.1f} MB")
