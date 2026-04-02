import numpy as np
import os
import glob
from ultralytics import YOLO
from paths import *

MODEL_PATH = 'cod/scooby_model6/weights/best.pt'
OUTPUT_DIR = 'bonus'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    model = YOLO(MODEL_PATH)
    
    img_paths = []
    img_paths.extend(glob.glob(os.path.join(VALIDATION_PATH, '*.jpg')))
    
    unknown_id = None
    for id, name in model.names.items():
        if name == 'unknown':
            unknown_id = id
            break

    class_data = {}
    for id, name in model.names.items():
        class_data[id] = {
            'name': name,
            'dets': [],   
            'scores': [], 
            'files': []   
        }
    
    results = model.predict(img_paths, conf=0.5, iou=0.5, stream=True, verbose=False)

    for r in results:
        file_name = os.path.basename(r.path)
        
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        
        for box, score, cls_id in zip(boxes, scores, classes):
            class_data[cls_id]['dets'].append(box)
            class_data[cls_id]['scores'].append(score)
            class_data[cls_id]['files'].append(file_name)
            
            if unknown_id is not None and cls_id != unknown_id:
                class_data[unknown_id]['dets'].append(box)
                class_data[unknown_id]['scores'].append(score)
                class_data[unknown_id]['files'].append(file_name)

    
    for cls_id, data in class_data.items():
        cls_name = data['name']
        
        if len(data['dets']) > 0:
            np_dets = np.array(data['dets'])
            np_scores = np.array(data['scores'])
            np_files = np.array(data['files'])
            
            np.save(os.path.join(OUTPUT_DIR, f"{"task1" if cls_name=="unknown" else "task2" }/detections_{cls_name if cls_name!="unknown" else "all_faces"}.npy"), np_dets)
            np.save(os.path.join(OUTPUT_DIR, f"{"task1" if cls_name=="unknown" else "task2" }/scores_{cls_name if cls_name!="unknown" else "all_faces"}.npy"), np_scores)
            np.save(os.path.join(OUTPUT_DIR, f"{"task1" if cls_name=="unknown" else "task2" }/file_names_{cls_name if cls_name!="unknown" else "all_faces"}.npy"), np_files)
            
if __name__ == "__main__":
    main()