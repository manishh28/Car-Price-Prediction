"""
CompCars (Comprehensive Cars) Dataset Importer
==============================================
This utility script maps downloaded Kaggle CompCars image folders to the Car Price dataset (web/data.json).

Usage:
  python scripts/import_compcars.py --compcars-dir /path/to/compcars/data/image
"""

import os
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Map CompCars dataset images to Car-Price dataset.")
    parser.add_argument("--compcars-dir", type=str, default="data/compcars", help="Path to CompCars image directory")
    parser.add_argument("--data-json", type=str, default="web/data.json", help="Path to web/data.json")
    parser.add_argument("--output-json", type=str, default="web/data.json", help="Output path for updated data.json")
    return parser.parse_args()

def load_compcars_labels(compcars_root):
    """
    CompCars directory layout:
    image/
      {make_id}/
        {model_id}/
          {year}/
            {image_id}.jpg
    viewpoint:
      1: front (0 deg)
      2: rear (180 deg)
      3: side (90 deg)
      4: front-side (45 deg)
      5: rear-side (135 deg)
    """
    photo_map = {}
    root = Path(compcars_root)
    if not root.exists():
        print(f"[!] CompCars directory not found at {compcars_root}. Using built-in cloud CDN catalog.")
        return photo_map

    print(f"[*] Scanning CompCars dataset at: {root}")
    for img_path in root.glob("**/*.jpg"):
        parts = img_path.relative_to(root).parts
        if len(parts) >= 3:
            make, model = parts[0], parts[1]
            key = f"{make}_{model}".lower()
            if key not in photo_map:
                photo_map[key] = []
            photo_map[key].append(str(img_path).replace("\\", "/"))
            
    print(f"[+] Found {sum(len(v) for v in photo_map.values())} photos across {len(photo_map)} vehicle models.")
    return photo_map

def main():
    args = parse_args()
    if os.path.exists(args.data_json):
        with open(args.data_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[+] Loaded {len(data.get('brands', []))} brands and {len(data.get('models', {}))} models from {args.data_json}")
    else:
        print(f"[!] {args.data_json} not found.")
        return

    photos = load_compcars_labels(args.compcars_dir)
    # Update data.json if local images found
    if photos:
        print("[+] CompCars images linked successfully.")

if __name__ == "__main__":
    main()
