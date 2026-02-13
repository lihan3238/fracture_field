from ultralytics import YOLO
import os
import shutil
import glob
import random

def split_dataset(train_ratio=0.8):
    """Automatically move 20% of data from train to val if val is empty."""
    base_dir = 'dataset'
    train_img_dir = os.path.join(base_dir, 'images', 'train')
    val_img_dir = os.path.join(base_dir, 'images', 'val')
    train_lbl_dir = os.path.join(base_dir, 'labels', 'train')
    val_lbl_dir = os.path.join(base_dir, 'labels', 'val')
    
    # Check if val is empty
    val_images = glob.glob(os.path.join(val_img_dir, '*.jpg'))
    if len(val_images) > 0:
        print(f"[INFO] Validation set has {len(val_images)} images. Skipping auto-split.")
        return

    # Get all training images
    train_images = glob.glob(os.path.join(train_img_dir, '*.jpg'))
    if len(train_images) == 0:
        print("[WARN] No training images found! Please collect data first with run_ai.py")
        return

    print(f"[INFO] Found {len(train_images)} training images. Moving 20% to validation...")
    
    # Calculate how many to move
    num_to_move = int(len(train_images) * (1 - train_ratio))
    if num_to_move < 1:
        num_to_move = 1
        
    to_move = random.sample(train_images, num_to_move)
    
    for img_path in to_move:
        # Move Image
        base_name = os.path.basename(img_path)
        new_img_path = os.path.join(val_img_dir, base_name)
        shutil.move(img_path, new_img_path)
        
        # Move Label
        label_name = base_name.replace('.jpg', '.txt')
        src_label = os.path.join(train_lbl_dir, label_name)
        dst_label = os.path.join(val_lbl_dir, label_name)
        
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
            
    print(f"[INFO] Moved {num_to_move} images to validation set.")

def train():
    # 0. Auto-balance dataset
    split_dataset()

    # 1. Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (n for Nano, fastest)

    # 2. Train usage
    # Ensure data.yaml is correct relative to execution
    yaml_path = os.path.abspath("dataset/data.yaml")
    
    print(f"Starting training using {yaml_path}")
    
    try:
        results = model.train(
            data=yaml_path, 
            epochs=100,
            imgsz=640, 
            device='cpu',
            plots=True,
            workers=0,
            fliplr=0.5,
            mosaic=0.5,
        )

        # 3. Validate
        print("Starting validation...")
        # Force validation on CPU to avoid "Invalid device id" if user has messy CUDA environment
        # or if simple "cpu" string doesn't propagate correctly in implicit calls.
        metrics = model.val(device='cpu') 
        print(f"mAP@50-95: {metrics.box.map}")

        # 4. Export
        print("Exporting model...")
        success = model.export(format="onnx")
        print("Training Complete. Model saved to runs/detect/train/weights/best.pt")
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    train()
