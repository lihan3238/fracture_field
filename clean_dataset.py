import os
import glob
import re

DATASET_ROOT = 'dataset'
LOG_DIR = 'logs'

def main():
    print("--- Dataset Processor (Hard Negative Mining) ---")
    print("Strategy: Files deleted from 'logs' will be marked as NEGATIVE (Background) for training.")
    print("          Files remaining in 'logs' are POSITIVE samples.")
    
    # 1. Get all valid IDs from logs and identify active batches
    valid_ids = set()
    active_prefixes = set() # Store YYYYMMDD_HH prefixes found in logs
    
    log_files = glob.glob(os.path.join(LOG_DIR, "*.jpg"))
    print(f"Found {len(log_files)} positive samples in logs.")
    
    for f in log_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 3:
            # Reconstruct ID (YYYYMMDD_HHMMSS_ffffff)
            file_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
            valid_ids.add(file_id)
            
            # Add prefix (YYYYMMDD_HH) to active list
            # ID format: 20260213_14xxxx...
            # Prefix len 11: 20260213_14
            if len(file_id) >= 11:
                active_prefixes.add(file_id[:11])

    print(f"Active batches in logs: {active_prefixes}")

    # 2. Process all images in dataset (train and val)
    # Recursively find all jpgs in dataset/images
    image_pattern = os.path.join(DATASET_ROOT, 'images', '**', '*.jpg')
    dataset_files = glob.glob(image_pattern, recursive=True)
    
    print(f"Scanning {len(dataset_files)} total images in dataset...")
    
    negative_count = 0
    positive_count = 0
    skipped_count = 0
    
    for img_path in dataset_files:
        basename = os.path.basename(img_path)
        file_id = os.path.splitext(basename)[0]
        
        # Check if this file belongs to a batch we are currently cleaning
        file_prefix = file_id[:11] if len(file_id) >= 11 else ""
        
        if file_prefix not in active_prefixes:
            # SAFEGUARD: This image belongs to an old run not present in logs.
            # Do not touch it.
            skipped_count += 1
            continue
        
        # Derive label path
        # images/train/foo.jpg -> labels/train/foo.txt
        label_path = img_path.replace(os.path.sep + 'images' + os.path.sep, os.path.sep + 'labels' + os.path.sep)
        label_path = os.path.splitext(label_path)[0] + '.txt'
        
        if not os.path.exists(os.path.dirname(label_path)):
            # Skip if directory structure is weird
            continue

        if file_id not in valid_ids:
            # CASE: NEGATIVE SAMPLE (User deleted from logs)
            # Action: Keep image, but EMPTY the label file
            # This teaches AI: "Nothing to see here"
            
            with open(label_path, 'w') as f:
                f.write("") # Empty file
                
            print(f"[NEGATIVE] {file_id} -> Labeled as background")
            negative_count += 1
        else:
            # CASE: POSITIVE SAMPLE
            # Action: Ensure label file exists (it should), leave it alone
            if os.path.exists(label_path):
                # Optional: Check if empty and restore? No, run_ai saved it correctly.
                positive_count += 1
            else:
                print(f"[WARN] Positive image {file_id} missing label file!")

    print("-" * 30)
    print(f"Processing Complete.")
    print(f"  Positives (Drones):     {positive_count}")
    print(f"  Negatives (Background): {negative_count}")
    print(f"  Skipped (Old Batches):  {skipped_count}")
    print("-" * 30)

if __name__ == "__main__":
    main()
