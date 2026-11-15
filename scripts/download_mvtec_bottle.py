"""
Download MVTec AD Bottle Dataset Script
Downloads all 209 defect-free training images and 20+ real defective images
directly from public high-speed mirror without PyTorch dependencies.
"""

import os
import sys
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://huggingface.co/datasets/foersben/mvtec-ad/resolve/main"
API_URL = "https://huggingface.co/api/datasets/foersben/mvtec-ad/tree/main"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mvtec_bottle")


def get_file_list(subpath: str):
    """Retrieve file list from Hugging Face API."""
    url = f"{API_URL}/{subpath}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        # return list of paths ending with .png
        return [item["path"] for item in data if item.get("path", "").endswith(".png")]


def download_single_file(remote_path: str, local_path: str):
    """Download a single file from the Hugging Face dataset."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return remote_path, True, "already exists"

    url = f"{BASE_URL}/{remote_path}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            content = resp.read()
            with open(local_path, "wb") as f:
                f.write(content)
        return remote_path, True, f"{len(content)} bytes"
    except Exception as e:
        return remote_path, False, str(e)


def main():
    print("=" * 60)
    print("MVTec AD Bottle Dataset Downloader")
    print(f"Target Directory: {DATA_DIR}")
    print("=" * 60)

    # 1. Fetch Normal Images (train/good)
    print("\n[1/4] Querying normal images (bottle/train/good)...")
    normal_files = get_file_list("bottle/train/good")
    print(f"Found {len(normal_files)} normal images in dataset.")

    # 2. Fetch Defective Images (broken_large, broken_small, contamination)
    print("\n[2/4] Querying defective images...")
    defect_categories = ["broken_large", "broken_small", "contamination"]
    defect_files_map = {}
    for cat in defect_categories:
        files = get_file_list(f"bottle/test/{cat}")
        defect_files_map[cat] = files
        print(f"  - {cat}: {len(files)} images")

    # 3. Prepare Download Tasks
    tasks = []
    # Normal images -> data/mvtec_bottle/train/good/
    normal_dir = os.path.join(DATA_DIR, "train", "good")
    for r_path in normal_files:
        filename = os.path.basename(r_path)
        dest = os.path.join(normal_dir, filename)
        tasks.append((r_path, dest))

    # Defect images -> data/mvtec_bottle/test/<category>/
    for cat, files in defect_files_map.items():
        cat_dir = os.path.join(DATA_DIR, "test", cat)
        for r_path in files:
            filename = os.path.basename(r_path)
            dest = os.path.join(cat_dir, filename)
            tasks.append((r_path, dest))

    print(f"\n[3/4] Downloading {len(tasks)} total images using multi-threading...")
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(download_single_file, r, d): (r, d) for r, d in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            r_path, ok, msg = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                print(f"  [ERROR] Failed to download {r_path}: {msg}")
            if i % 25 == 0 or i == len(tasks):
                print(f"  Progress: {i}/{len(tasks)} completed ({success_count} success, {fail_count} failed)")

    # 4. Create a dedicated folder with exactly 20 diverse defective images for quick testing
    print("\n[4/4] Creating curated 'defective_20' sample folder...")
    curated_dir = os.path.join(DATA_DIR, "defective_20")
    os.makedirs(curated_dir, exist_ok=True)

    # 7 broken_large, 7 broken_small, 6 contamination = 20 images
    selection = (
        [(f, "broken_large") for f in defect_files_map["broken_large"][:7]] +
        [(f, "broken_small") for f in defect_files_map["broken_small"][:7]] +
        [(f, "contamination") for f in defect_files_map["contamination"][:6]]
    )

    for idx, (r_path, cat) in enumerate(selection, 1):
        filename = os.path.basename(r_path)
        src = os.path.join(DATA_DIR, "test", cat, filename)
        dst_name = f"defect_{idx:02d}_{cat}_{filename}"
        dst = os.path.join(curated_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Summary verification
    normal_downloaded = len(os.listdir(normal_dir)) if os.path.exists(normal_dir) else 0
    curated_count = len(os.listdir(curated_dir)) if os.path.exists(curated_dir) else 0
    total_defects = sum(len(os.listdir(os.path.join(DATA_DIR, "test", cat))) for cat in defect_categories if os.path.exists(os.path.join(DATA_DIR, "test", cat)))

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETED!")
    print(f"  - Normal Images in 'train/good': {normal_downloaded} (Expected: 209)")
    print(f"  - Defective Images in 'test/': {total_defects} across 3 categories")
    print(f"  - Curated 'defective_20/' folder: {curated_count} images ready for testing")
    print(f"Location: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
