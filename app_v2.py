import logging
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Constants
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
ALLOWED_FORMATS = {'.jpg', '.jpeg', '.png'}
MIN_IMAGE_SIZE = (50, 50)  # Minimum required resolution

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Load CNN Model (assuming a pre-trained model is available)
CNN_MODEL_PATH = "face_detection_cnn.h5"  # Replace with actual path if available
cnn_model = load_model(CNN_MODEL_PATH) if os.path.exists(CNN_MODEL_PATH) else None

def is_valid_image(image_path):
    """Check if image format, quality, and size are valid."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ALLOWED_FORMATS:
        logging.error("Unsupported file format. Only JPEG and PNG are allowed.")
        return False
    
    image = cv2.imread(image_path)
    if image is None:
        logging.error("Corrupted image file or unreadable format.")
        return False
    
    if image.shape[0] < MIN_IMAGE_SIZE[0] or image.shape[1] < MIN_IMAGE_SIZE[1]:
        logging.error("Image resolution is too low.")
        return False
    
    return True

def preprocess_image(image_path):
    """Preprocess the image: Resize, grayscale, and normalize."""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_image = gray_image / 255.0  # Normalize pixel values
    return gray_image, normalized_image

def detect_faces(image_path):
    """Detect faces using Haar Cascade and CNN if available."""
    try:
        if not is_valid_image(image_path):
            return None, []
        
        gray_image, _ = preprocess_image(image_path)
        image = cv2.imread(image_path)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if cnn_model:
            faces = refine_with_cnn(image, faces)
        
        logging.info(f"Faces detected: {len(faces)}")
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return image, faces
    except Exception as e:
        logging.error(f"Error during face detection: {e}")
        return None, []

def refine_with_cnn(image, faces):
    """Refine Haar Cascade results with CNN model to reduce false positives."""
    refined_faces = []
    for (x, y, w, h) in faces:
        face_roi = cv2.resize(image[y:y+h, x:x+w], (64, 64))
        face_roi = np.expand_dims(face_roi, axis=0) / 255.0
        prediction = cnn_model.predict(face_roi)
        if prediction > 0.5:
            refined_faces.append((x, y, w, h))
    return refined_faces

def save_detected_image(image, output_path):
    """Save image with detected faces."""
    try:
        cv2.imwrite(output_path, image)
        logging.info(f"Image saved successfully at {output_path}")
    except Exception as e:
        logging.error(f"Error while saving image: {e}")

def batch_process_images(input_folder, output_folder):
    """Process multiple images from a folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        detected_image, faces = detect_faces(image_path)
        if detected_image is not None:
            save_detected_image(detected_image, output_path)

# Example usage
if __name__ == "__main__":
    image_path = 'input_image.jpg'  # Replace with actual image path
    output_path = 'output_image.jpg'
    
    detected_image, faces = detect_faces(image_path)
    if detected_image is not None:
        save_detected_image(detected_image, output_path)
    
    # Batch processing example
    batch_process_images('input_images', 'output_images')
