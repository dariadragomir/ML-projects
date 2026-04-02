from helpers import *
import os
import numpy as np
import cv2 as cv
from paths import *
from constants import *
import torch

imgs = np.array(import_images(VALIDATION_PATH + "*.jpg", True))
if len(imgs) == 0:
    print("Error: No images found. Check VALIDATION_PATH.")
    exit()
print(f"Loaded {len(imgs)} images.")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

modelVelma = torch.load('cod/Models/CNNvelma5sigmoid.pth', map_location=device, weights_only=False)
modelDaphne = torch.load('cod/Models/CNNDaphne5_sigmoid.pth', map_location=device, weights_only=False)
modelFred = torch.load('cod/Models/CNNfred5sigmoid.pth', map_location=device, weights_only=False)
modelShaggy = torch.load('cod/Models/CNNshaggy5sigmoid.pth', map_location=device, weights_only=False)
modelUnknown = torch.load('cod/Models/CNNunknown5sigmoid.pth', map_location=device, weights_only=False)

models = { "velma": modelVelma, "daphne": modelDaphne, "fred":modelFred, "shaggy":modelShaggy, "unknown": modelUnknown}


detections_per_image = { l : {i: {'boxes': [], 'scores': []} for i in range(len(imgs))} for l in LABELS }
for scale in SCALES:
    new_width = int(imgs.shape[2] * scale)
    new_height = int(imgs.shape[1] * scale)
    resized_imgs = torch.from_numpy(np.array([cv.resize(imgs[i], (new_width, new_height)) for i in range(imgs.shape[0])])) / 255.0
    #print(resized_imgs.shape)
    
    for label in LABELS:

        win_h = WINDOW_SIZES[label][0]
        win_w = WINDOW_SIZES[label][1]
        
        windows = resized_imgs.unfold(1, win_h, step=STRIDE).unfold(2, win_w, step=STRIDE)
        
        n_rows = windows.shape[1]
        n_cols = windows.shape[2]
        num_windows_per_image = n_rows * n_cols
        
        windows = windows.reshape(-1, 3, win_h, win_w)
        
        dataset = torch.utils.data.TensorDataset(windows)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        all_pred = []
        all_scores = []
        with torch.no_grad():
            for batch in dataloader:
                im = batch[0].to(device)
                
                outputs = models[label](im)
                
                preds = (torch.nn.functional.sigmoid(outputs)>=SCORE_THRESH[label]).to(dtype=torch.int32)
                
                all_pred.append(preds.squeeze())
                all_scores.append(torch.sigmoid(outputs).squeeze())
                
            all_pred = torch.cat(all_pred)
            all_scores = torch.cat(all_scores)
            
            indices = torch.nonzero(all_pred == 1).squeeze()
            if indices.shape == torch.Size([]):
                continue
            scores = all_scores[indices]
            print(indices)
            
            image_indexes = indices // num_windows_per_image
            window_id = indices % num_windows_per_image
            row = window_id // n_cols
            column = window_id% n_cols
            x_left = ((column*STRIDE)/scale).to(dtype=torch.int32)
            y_left = ((row*STRIDE)/scale).to(dtype=torch.int32)
            x_right = x_left + WINDOW_SIZES[label][1] // scale
            y_right = y_left + WINDOW_SIZES[label][0] // scale
            
            boxes = torch.stack((x_left, y_left, x_right, y_right), dim=1)
            boxes_np = boxes.cpu().numpy()
            scores_np = scores.cpu().numpy()
            image_indexes_np = image_indexes.cpu().numpy()
            
            for i in range(len(boxes_np)):
                img_idx = image_indexes_np[i]
                if label == "unknown":
                    detections_per_image[label][img_idx]['boxes'].append(boxes_np[i])
                    detections_per_image[label][img_idx]['scores'].append(scores_np[i])
                else:
                    detections_per_image[label][img_idx]['boxes'].append(boxes_np[i])
                    detections_per_image[label][img_idx]['scores'].append(scores_np[i])
                    detections_per_image["unknown"][img_idx]['boxes'].append(boxes_np[i])
                    detections_per_image["unknown"][img_idx]['scores'].append(scores_np[i])
            
        
dets_final = { l :[] for l in LABELS}
scores_final = { l : [] for l in LABELS}
idxs_final = { l: [] for l in LABELS}
for label in LABELS:
    for img_idx, data in detections_per_image[label].items():
        if len(data['boxes']) == 0:
            continue
        #print(img_idx)
        image_detections = np.array(data['boxes'])
        image_scores = np.array(data['scores'])
        
        final_dets, final_scores = non_maximal_suppression(
            image_detections, 
            image_scores, 
            imgs[0].shape
        )
        
        for d,s in zip(final_dets, final_scores):
            x_min, y_min, x_max, y_max = d
            w = x_max - x_min
            h = y_max - y_min
            
            dets_final[label].append(d)
            scores_final[label].append(s)
            idxs_final[label].append(img_idx)
        
    np.save(f"./{"task1" if label=="unknown" else "task2" }/detections_{label if label!="unknown" else "all_faces"}", np.array(dets_final[label]))
    np.save(f"./{"task1" if label=="unknown" else "task2" }/scores_{label if label!="unknown" else "all_faces"}", np.array(scores_final[label]))
    np.save(f"./{"task1" if label=="unknown" else "task2" }/file_names_{label if label!="unknown" else "all_faces"}", np.array([f"{idx+1:04d}.jpg" for idx in idxs_final[label]]))