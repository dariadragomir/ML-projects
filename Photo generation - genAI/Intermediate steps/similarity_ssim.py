from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

def calculate_similarity_percentage(imageA_path, imageB_path):
    imageA = cv2.imread(imageA_path)
    imageB = cv2.imread(imageB_path)
    
    imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))
    
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    ssim_index = ssim(imageA, imageB)
    similarity_percentage = ssim_index * 100
    
    return similarity_percentage

similarity = calculate_similarity_percentage("/Users/dariadragomir/FMI-Unibuc/DAW_Django/Laboratoare/lab1/final_image.png", "/Users/dariadragomir/AI_siemens/gen photos/final_image_2.png")
print(f"Similarity: {similarity:.2f}%")
