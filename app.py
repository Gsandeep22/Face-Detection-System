import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import requests
import base64
from datetime import datetime
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['MODEL_FOLDER'] = 'models'


db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure upload & processed folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    images = db.relationship('Image', backref='user', lazy=True)

# Image Model
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.String(255), nullable=False)
    processed_path = db.Column(db.String(255), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Home Route
@app.route('/')
def index():
    if current_user.is_authenticated:
        user_images = Image.query.filter_by(user_id=current_user.id).all()
    else:
        user_images = []
    return render_template('index.html', images=user_images)

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')

    return render_template('login.html')

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# Haar Cascade Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Pretrained Model URLs
MODEL_URL = "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/"
PROTO_TXT = "deploy.prototxt"
MODEL_WEIGHTS = "res10_300x300_ssd_iter_140000.caffemodel"

proto_path = os.path.join(app.config['MODEL_FOLDER'], PROTO_TXT)
model_path = os.path.join(app.config['MODEL_FOLDER'], MODEL_WEIGHTS)

# Download Model if Not Exists
def download_file(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded: {save_path}")

download_file(MODEL_URL + PROTO_TXT, proto_path)
download_file(MODEL_URL + MODEL_WEIGHTS, model_path)

# Load DNN Model
face_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

def iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = w1 * h1
    box2Area = w2 * h2
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

@app.route('/capture', methods=['POST'])
@login_required
def capture():
    """Handles image capture from webcam, processes face detection, and saves to DB."""
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image data received"}), 400

    try:
        # Decode base64 image
        image_data = data['image'].split(",")[1]
        image_bytes = base64.b64decode(image_data)

        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]  # Generate a short random ID
        original_filename = f"captured_{timestamp}_{unique_id}.jpg"
        processed_filename = f"processed_{timestamp}_{unique_id}.jpg"

        # Secure file paths
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(original_filename))
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], secure_filename(processed_filename))

        # Save the image
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        # Read Image
        image = cv2.imread(file_path)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Face Detection (Haar Cascade)
        faces_haar = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        # Face Detection (DNN)
        blob = cv2.dnn.blobFromImage(image_rgb, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        detected_faces = []

        # Process DNN Faces
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x_max, y_max) = box.astype("int")
                detected_faces.append((x, y, x_max - x, y_max - y))

        # Add Haar faces
        for (x, y, w, h) in faces_haar:
            detected_faces.append((x, y, w, h))

        # Remove duplicate detections using IoU
        unique_faces = []
        for face in detected_faces:
            if all(iou(face, unique_face) < 0.5 for unique_face in unique_faces):
                unique_faces.append(face)

        # Draw bounding boxes
        for (x, y, w, h) in unique_faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Save processed image
        cv2.imwrite(processed_path, image)

        # Save image record to DB
        new_image = Image(
            file_path=file_path,
            processed_path=processed_path,
            user_id=current_user.id
        )
        db.session.add(new_image)
        db.session.commit()

        return jsonify({"processed_path": f"/processed/{processed_filename}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Predict (Face Detection) Route
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handles image upload, detects faces using Haar + DNN, removes duplicates, and saves results."""
    
    if 'file' not in request.files:
        flash('No file uploaded!', 'danger')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file!', 'danger')
        return redirect(url_for('index'))

    try:
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]  # Generate short random ID
        original_filename = f"uploaded_{timestamp}_{unique_id}.jpg"
        processed_filename = f"processed_{timestamp}_{unique_id}.jpg"

        # Secure file paths
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(original_filename))
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], secure_filename(processed_filename))

        # Save the uploaded image
        file.save(file_path)

        # Read Image
        image = cv2.imread(file_path)
        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for Haar

        # Face Detection (Haar Cascade)
        faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Face Detection (DNN)
        blob = cv2.dnn.blobFromImage(image_rgb, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        detected_faces = []

        # Process DNN Faces
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x_max, y_max) = box.astype("int")
                detected_faces.append((x, y, x_max - x, y_max - y))

        # Add Haar faces
        for (x, y, w, h) in faces_haar:
            detected_faces.append((x, y, w, h))

        # Remove duplicate detections using IoU
        unique_faces = []
        for face in detected_faces:
            if all(iou(face, unique_face) < 0.5 for unique_face in unique_faces):
                unique_faces.append(face)

        # Draw bounding boxes
        for (x, y, w, h) in unique_faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Save processed image
        cv2.imwrite(processed_path, image)

        # Save image record to DB
        new_image = Image(file_path=file_path, processed_path=processed_path, user_id=current_user.id)
        db.session.add(new_image)
        db.session.commit()

        flash('Face detection completed successfully!', 'success')
        return redirect(url_for('index'))

    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('index'))
    
# Serve Processed Images
@app.route('/processed/<filename>')
@login_required
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# Run the Flask app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
