import cv2
import os
import logging

# Initialize logging to store the results
logging.basicConfig(level=logging.INFO)

# Path to the pre-trained Haar Cascade Classifier
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Function to load the image and detect faces
def detect_faces(image_path):
    try:
        # Load the Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

        # Read the image
        image = cv2.imread(image_path)

        if image is None:
            logging.error("Image could not be loaded. Please check the file path.")
            return None

        # Convert the image to grayscale (Haar Cascade works better with grayscale images)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Log the number of faces detected
        logging.info(f"Faces detected: {len(faces)}")

        # Draw bounding boxes around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Return the image with bounding boxes drawn around detected faces
        return image, faces

    except Exception as e:
        logging.error(f"An error occurred during face detection: {e}")
        return None

# Function to save the image with detected faces
def save_detected_image(image, output_path):
    try:
        cv2.imwrite(output_path, image)
        logging.info(f"Image saved successfully at {output_path}")
    except Exception as e:
        logging.error(f"An error occurred while saving the image: {e}")

# Example usage:
if __name__ == "__main__":
    image_path = 'input_image.jpg'  # Replace with the path to your image
    output_path = 'output_image.jpg'  # Output path where the image with bounding boxes will be saved

    detected_image, faces = detect_faces(image_path)
    
    if detected_image is not None:
        save_detected_image(detected_image, output_path)
