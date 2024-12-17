import numpy as np
import cv2  
from tensorflow.keras.models import load_model

def preprocess_image(image_path, img_shape):
    """
    Preprocesses an image: resizes it to the target shape and normalizes pixel values.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(img, (img_shape[0], img_shape[1]))  
    img = img.astype("float32") / 255.0  
    img = np.expand_dims(img, axis=-1) 
    return img

def compare_images(image1_path, image2_path, model, img_shape=(28, 28, 1), threshold=0.5):
    """
    Compares two images to check if they belong to the same class.

    Args:
    - image1_path: Path to the first image.
    - image2_path: Path to the second image.
    - model: The trained Siamese network model.
    - img_shape: Target shape for model input, default is (28, 28, 1).
    - threshold: Similarity threshold for classifying as "same" or "different".

    Returns:
    - A similarity score between 0 and 1.
    - Classification of 'Same Class' or 'Different Class' based on threshold.
    """
    img1 = preprocess_image(image1_path, img_shape)
    img2 = preprocess_image(image2_path, img_shape)

    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    similarity_score = model.predict([img1, img2])[0][0]

    if similarity_score >= threshold:
        classification = "Same Class"
    else:
        classification = "Different Class"

    return similarity_score, classification
