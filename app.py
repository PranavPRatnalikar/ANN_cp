from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import firebase_admin
from firebase_admin import credentials, db
import face_recognition
import numpy as np
import cv2
import dlib
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

CORS(app, supports_credentials=True, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configure CORS to allow requests from React frontend
CORS(app, resources={r"/*": {"origins": "*"}})

# Firebase setup
current_directory = os.path.dirname(os.path.abspath(__file__))
firebase_cred_path = os.path.join(current_directory, "galaxy-unhilater-firebase-adminsdk-6diyp-a486ede4b4.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_cred_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://galaxy-unhilater-default-rtdb.firebaseio.com'
    })

# Dlib setup for face alignment
shape_predictor_path = "FaceNet.dat"
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)

def add_to_firebase(prn, name, image_encoding):
    current_date = datetime.now().strftime("%Y-%m-%d")
    ref = db.reference(f'FaceData/{current_date}/{prn}')
    data = {
        'name': name,
        'encoding': image_encoding.tolist(),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    ref.set(data)
    return True

def get_known_faces():
    current_date = datetime.now().strftime("%Y-%m-%d")
    ref = db.reference(f'FaceData/{current_date}')
    face_data = ref.get()
    known_encodings, known_prns, known_names = [], [], []

    if face_data:
        for prn, data in face_data.items():
            if 'encoding' in data:
                known_encodings.append(np.array(data['encoding']))
                known_prns.append(prn)
                known_names.append(data['name'])
    return known_encodings, known_prns, known_names

# Basic route to test if server is running
@app.route('/')
def home():
    return jsonify({"message": "Flask server is running"}), 200

# Test route
@app.route('/test')
def test():
    return jsonify({"message": "Test endpoint working"}), 200

@app.route('/add_student', methods=['POST'])
def add_student():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
            
        data = request.form
        prn = data.get('prn')
        name = data.get('name')
        
        if not prn or not name:
            return jsonify({"error": "PRN and name are required"}), 400
            
        image_file = request.files['image']
        
        # Process image
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400
            
        faces = detector(img)
        if not faces:
            return jsonify({"error": "No face detected in the image"}), 400
            
        shape = shape_predictor(img, faces[0])
        aligned_face = dlib.get_face_chip(img, shape, size=256)
        encoding = face_recognition.face_encodings(aligned_face)[0]
        
        if add_to_firebase(prn, name, encoding):
            return jsonify({"message": "Student added successfully"}), 200
        else:
            return jsonify({"error": "Failed to add student to database"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/take_attendance', methods=['POST'])
def take_attendance():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
            
        image_file = request.files['image']
        
        # Process image
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400
            
        known_encodings, known_prns, known_names = get_known_faces()
        faces = detector(img)
        
        if not faces:
            return jsonify({"error": "No faces detected in the image"}), 400
            
        results = []
        for face in faces:
            shape = shape_predictor(img, face)
            aligned_face = dlib.get_face_chip(img, shape, size=256)
            face_encoding = face_recognition.face_encodings(aligned_face)[0]
            
            if known_encodings:
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                confidence = (1 - face_distances[best_match_index]) * 100
                
                if confidence >= 50:
                    results.append({
                        "prn": known_prns[best_match_index],
                        "name": known_names[best_match_index],
                        "confidence": confidence
                    })
        
        return jsonify({"attendance": results}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app on host='0.0.0.0' to make it accessible from other machines
    app.run(host='0.0.0.0', port=5000, debug=True)