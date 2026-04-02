import os
import shutil
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from paths import *

CLASS_MAPPING = {
    'daphne': 0,
    'fred': 1,
    'shaggy': 2,
    'velma': 3,
    'unknown': 4
}

FOLDERS = ['daphne', 'fred', 'shaggy', 'velma']
IMG_WIDTH = 480
IMG_HEIGHT = 360

OUTPUT_DIR = Path('scooby_doo')

def convert_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    x_center /= img_w
    y_center /= img_h
    width /= img_w
    height /= img_h
    
    return x_center, y_center, width, height

def main():
    for split in ['train', 'val']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    all_data = []

    for folder in FOLDERS:
        annotation_file = os.path.join(PATH_ANNOTATIONS, f"{folder}_annotations.txt")
        
        if not os.path.exists(annotation_file):
            print(f"Warning: {annotation_file} not found, skipping...")
            continue
            
        
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            
        img_annotations = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6: continue
            
            filename = parts[0]
            
            filename = os.path.basename(filename) 
            
            coords = list(map(int, parts[1:5])) # x1, y1, x2, y2
            label_text = parts[5]
            
            if label_text not in CLASS_MAPPING:
                print(f"Skipping unknown class: {label_text}")
                continue
                
            if filename not in img_annotations:
                img_annotations[filename] = []
            img_annotations[filename].append((coords, CLASS_MAPPING[label_text]))

        files = list(img_annotations.keys())
        train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

        for filename in files:
            split = 'train' if filename in train_files else 'val'
            
            src_img_path = Path(PATH_ANNOTATIONS + folder) / filename
            
            if not src_img_path.exists():
                src_img_path = src_img_path.with_suffix('.jpg') 

            dst_img_path = OUTPUT_DIR / 'images' / split / f"{folder}_{filename}"
            dst_label_path = OUTPUT_DIR / 'labels' / split / f"{folder}_{filename}".replace(src_img_path.suffix, '.txt')

            shutil.copy(src_img_path, dst_img_path)

            with open(dst_label_path, 'w') as out_f:
                for (coords, class_id) in img_annotations[filename]:
                    xmin, ymin, xmax, ymax = coords
                    xc, yc, w, h = convert_to_yolo(xmin, ymin, xmax, ymax, IMG_WIDTH, IMG_HEIGHT)
                    out_f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    main()