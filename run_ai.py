from ultralytics import YOLO
import cv2
import mss
import numpy as np
import pyautogui
import time
import os
import datetime
import glob

def get_latest_model():
    """Find the most recent trained model in runs/detect/."""
    # Base directory where YOLO saves runs
    base_dir = os.path.join('runs', 'detect')
    if not os.path.exists(base_dir):
        return None
        
    # Find all 'train*' directories
    run_dirs = glob.glob(os.path.join(base_dir, 'train*'))
    
    # Sort by modification time (newest first)
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    
    for run_dir in run_dirs:
        # Check specific structure "weights/best.pt"
        model_path = os.path.join(run_dir, 'weights', 'best.pt')
        if os.path.exists(model_path):
            return model_path
            
    # Fallback to standard convention if auto-discovery fails
    # But now we use 'runs/detect/train' as the training script uses strictly
    # increasing names like train, train2, train3...
    
    # Try the explicit first one
    default_path = 'runs/detect/train/weights/best.pt'
    if os.path.exists(default_path):
        return default_path
        
    return None

# Configuration
CONF_THRESHOLD = 0.5    # Confidence needed to act
COOLDOWN = 1.0          # Seconds between actions
MODEL_PATH = get_latest_model()  # Dynamically find latest model
COLLECT_DATA = True     # If True, save detections for future training
DATASET_ROOT = 'dataset'
LOG_ROOT = 'logs'       # For human debugging (save crops)

# Maintain logs directory
if not os.path.exists(LOG_ROOT):
    os.makedirs(LOG_ROOT)

# Template Matching Config (Fallback)
MATCH_THRESHOLD = 0.60
SCALE_DESC = (0.8, 1.2, 5)
TEMPLATE_BASE = 'assets'

# Class Map (For saving labels)
# 0: drone_off (both left and right)
# 1: drone_on  (both left and right)
CLASS_MAP = {
    'drone_off': 0,
    'drone_on': 1
}

# Ensure directories exist if collecting data
if COLLECT_DATA:
    os.makedirs(os.path.join(DATASET_ROOT, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, 'labels', 'train'), exist_ok=True)

def load_templates():
    templates = {}
    
    # Load basic templates (l_off, l_on)
    # Automatically generate flipped versions (r_off, r_on)
    base_names = ['l_off', 'l_on']
    
    for name in base_names:
        path = os.path.join(TEMPLATE_BASE, f"{name}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Store original (Left)
                # Map internal template name to generalized class name for easier processing later
                # We use a tuple tag: (display_name, target_class_name)
                # Actually, let's just keep specific names for template matching, map later
                templates[name] = img
                
                # Generate Flipped (Right)
                flipped_img = cv2.flip(img, 1)
                r_name = name.replace('l_', 'r_')
                templates[r_name] = flipped_img
                print(f"[INIT] Loaded {name} and generated {r_name}")
    return templates

def get_template_matches(screen_gray, templates):
    """Fallback detection using template matching."""
    matches = []
    # Generate scales
    scales = np.linspace(SCALE_DESC[0], SCALE_DESC[1], SCALE_DESC[2])
    
    for t_name, tpl in templates.items():
        t_h, t_w = tpl.shape[:2]
        
        # Determine generalized class
        if 'off' in t_name:
            gen_cls = 'drone_off'
        else:
            gen_cls = 'drone_on'
            
        for scale in scales:
            resize_w, resize_h = int(t_w * scale), int(t_h * scale)
            if resize_w > screen_gray.shape[1]: continue
            
            resized = cv2.resize(tpl, (resize_w, resize_h))
            res = cv2.matchTemplate(screen_gray, resized, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= MATCH_THRESHOLD)
            
            for pt in zip(*loc[::-1]):
                matches.append({
                    'x': pt[0], 'y': pt[1],
                    'w': resize_w, 'h': resize_h,
                    'score': res[pt[1], pt[0]],
                    'cls': gen_cls  # Use generalized name (drone_off/on)
                })
    return nms(matches)

def nms(boxes, overlap_thresh=0.3):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda x: x['score'], reverse=True)
    pick = []
    while boxes:
        current = boxes.pop(0)
        pick.append(current)
        survivors = []
        cx = current['x'] + current['w']/2
        cy = current['y'] + current['h']/2
        for other in boxes:
            ox = other['x'] + other['w']/2
            oy = other['y'] + other['h']/2
            dist = np.sqrt((cx-ox)**2 + (cy-oy)**2)
            if dist > (current['w'] + other['w'])/2 * overlap_thresh:
                survivors.append(other)
        boxes = survivors
    return pick

def save_debug_crop(img_bgr, d, batch_id):
    """Save a cropped image to logs folder for easy viewing."""
    try:
        x, y, w, h = int(d['x']), int(d['y']), int(d['w']), int(d['h'])
        # Add padding
        pad = 20
        h_src, w_src = img_bgr.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_src, x + w + pad)
        y2 = min(h_src, y + h + pad)
        
        crop = img_bgr[y1:y2, x1:x2].copy()
        
        # Draw box on crop
        cv2.rectangle(crop, (x - x1, y - y1), (x - x1 + w, y - y1 + h), (0, 255, 0), 2)
        
        # Filename: batch_id + info.jpg
        # batch_id is already unique timestamp
        filename = f"{batch_id}_{d['cls']}_{d['score']:.2f}.jpg"
        cv2.imwrite(os.path.join(LOG_ROOT, filename), crop)
    except Exception as e:
        print(f"Log error: {e}")

def save_training_data(img_bgr, formatted_detections, batch_id):
    """Save image and YOLO format label."""
    if not formatted_detections: return
    
    # Save Image using batch_id
    cv2.imwrite(f"{DATASET_ROOT}/images/train/{batch_id}.jpg", img_bgr)
    
    # Save Label
    h, w = img_bgr.shape[:2]
    with open(f"{DATASET_ROOT}/labels/train/{batch_id}.txt", 'w') as f:
        for d in formatted_detections:
            # YOLO format: class x_center y_center width height (normalized)
            dw = 1./w
            dh = 1./h
            x_center = (d['x'] + d['w']/2.0) * dw
            y_center = (d['y'] + d['h']/2.0) * dh
            width = d['w'] * dw
            height = d['h'] * dh
            
            cls_id = CLASS_MAP.get(d['cls'], -1)
            if cls_id != -1:
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"[DATA] Saved {batch_id} with {len(formatted_detections)} labels.")

def main():
    use_yolo = False
    model = None
    templates = {}

    if MODEL_PATH and os.path.exists(MODEL_PATH):
        print(f"Loading custom model from {MODEL_PATH}...")
        try:
            model = YOLO(MODEL_PATH)
            use_yolo = True
        except Exception as e:
            print(f"Error loading YOLO: {e}")
    else:
        print(f"Model not found at {MODEL_PATH}")
    
    if not use_yolo:
        print("Model not found or failed. Using TEMPLATE MATCHING fallback.")
        templates = load_templates()
        if not templates:
            print("Error: No templates found either!")
            return

    print(f"Monitoring... (Mode: {'YOLO' if use_yolo else 'Template Matching'})")
    print(f"Data Collection: {'ON' if COLLECT_DATA else 'OFF'}")
    
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        
        try:
            while True:
                # Capture
                img = np.array(sct.grab(monitor))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                detections = []
                
                if use_yolo:
                    results = model(img_bgr, verbose=False, conf=CONF_THRESHOLD)
                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            # Handle different class lists (YOLOv8 default vs ours)
                            if cls_id < len(model.names):
                                name = model.names[cls_id]
                                detections.append({
                                    'x': box.xyxy[0][0].item(), # Left
                                    'y': box.xyxy[0][1].item(), # Top
                                    'w': box.xywh[0][2].item(), # Width
                                    'h': box.xywh[0][3].item(), # Height
                                    'score': float(box.conf[0]),
                                    'cls': name
                                })
                else:
                    # Template Matching Fallback
                    candidates = get_template_matches(img_gray, templates)
                    detections = nms(candidates)
                
                # --- AUTO-LOGGING ---
                if COLLECT_DATA and detections:
                    # Generate a unique ID for this 'frame event'
                    batch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                    # 1. Save FULL image for Training (YOLO needs context)
                    save_training_data(img_bgr, detections, batch_id)
                    
                    # 2. Save CROPPED image for Debugging (Logs)
                    # Just save the first/best one to avoid spamming
                    save_debug_crop(img_bgr, detections[0], batch_id)
                    
                    # Prevent spamming 100 images/sec
                    time.sleep(1.0)
                
                # --- ACTION ---
                for d in detections:
                    if d['cls'] == 'drone_off':
                        print(f"[ACTION] Waking {d['cls']} (Conf: {d['score']:.2f})")
                        cx = d['x'] + d['w']/2
                        cy = d['y'] + d['h']/2
                        
                        screen_x = monitor['left'] + cx
                        screen_y = monitor['top'] + cy
                        
                        pyautogui.moveTo(screen_x, screen_y)
                        
                        # Add a tiny movement or click if needed?
                        # pyautogui.click() 
                        
                        # Once we act, break so we don't jump around crazily in one frame
                        break 
                
                # Simple loop delay if no data collected
                if not COLLECT_DATA:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
