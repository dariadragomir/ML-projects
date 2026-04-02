from helpers import *
import os
import numpy as np
import cv2 as cv
from paths import *
from constants import *

faces = {name: [] for name in SCOOBY_CLASSES}
faces_zoomed_out = {name: [] for name in SCOOBY_CLASSES}
mean_aspect_ratios = {name: 0 for name in SCOOBY_CLASSES}
number_of_examples = {name: 0 for name in SCOOBY_CLASSES}
mean_size = {name: [0, 0] for name in SCOOBY_CLASSES}

for folder_name in FOLDERS:
    path_to_imgs = os.path.join(PATH, folder_name, "*.jpg") 
    
    imgs = import_images(path_to_imgs, True) 
    
    # format: nume_imagine xmin ymin xmax ymax nume_personaj 
    annotation_file = os.path.join(PATH_ANNOTATIONS, folder_name + "_annotations.txt")
    
    f = open(annotation_file)
    lines = f.readlines()
    f.close()

    for line in lines:
        comp = line.split()
        
        img_no = int(comp[0].split('.')[0]) 
        
        x_min = int(comp[1])
        y_min = int(comp[2])
        x_max = int(comp[3])
        y_max = int(comp[4])
        label = comp[5].lower()

        if label not in faces:
            print("label not in faces")
            continue

        #y_min:y_max, x_min:x_max
        face_crop = imgs[img_no - 1][y_min : y_max + 1, x_min: x_max + 1]
        new_xmin = int(0.95*x_min)
        new_ymin = int(0.95*y_min)
        new_xmax = int(1.05*x_max)
        new_ymax = int(1.05*y_max)
        if new_xmin <= 0:
            new_xmin =x_min
        if new_ymin <= 0:
            new_ymin = y_min
        if new_ymax >= imgs[img_no - 1].shape[0]:
            new_ymax = y_max
        if new_xmax >= imgs[img_no - 1].shape[1]:
            new_xmax = x_max
        face_zoom_out_crop = imgs[img_no - 1][new_ymin : new_ymax + 1, new_xmin: new_xmax + 1]
        resized_face = cv.resize(face_crop, (WINDOW_SIZES[label][1], WINDOW_SIZES[label][0]))
        faces[label].append(resized_face)
        resized_face_zoomed_out = cv.resize(face_zoom_out_crop, (WINDOW_SIZES[label][1], WINDOW_SIZES[label][0]))
        faces_zoomed_out[label].append(resized_face_zoomed_out)

        height = y_max - y_min
        width = x_max - x_min
        
        if width > 0:
            mean_aspect_ratios[label] += height / width
            
        mean_size[label][0] += height
        mean_size[label][1] += width
        number_of_examples[label] += 1

print("\nStatistics")
for char in SCOOBY_CLASSES:
    if number_of_examples[char] > 0:
        save_path = os.path.join(FACES_PATH, char)
        np.save(save_path, faces[char])
        np.save(save_path+'_zoomed_out', faces_zoomed_out[char])
        avg_aspect = mean_aspect_ratios[char] / number_of_examples[char]
        avg_height = mean_size[char][0] / number_of_examples[char]
        avg_width = mean_size[char][1] / number_of_examples[char]
        
        print(f"{char.capitalize()}:")
        print(f"  Count: {number_of_examples[char]}")
        print(f"  Avg Aspect Ratio: {avg_aspect:.2f}")
        print(f"  Avg Height: {avg_height:.2f}")
        print(f"  Avg Width: {avg_width:.2f}")
    else:
        print(f"{char}: No examples found.")