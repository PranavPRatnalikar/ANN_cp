import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import face_recognition
import numpy as np
import cv2
import dlib

# Firebase setup (initialize using your Firebase credentials)
current_directory = os.path.dirname(os.path.abspath(__file__))
firebase_cred_path = os.path.join(current_directory, "espiotproject1-firebase-adminsdk-856yr-1b9ae0c516.json")

# Check if Firebase is already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_cred_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://espiotproject1-default-rtdb.firebaseio.com'
    })

# Reference to the root of your Firebase Realtime Database
ref = db.reference('/')

# Dlib model for face detection and landmarking
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)

# Function to load and encode an image
def load_and_encode(image_file):
    try:
        image_bytes = image_file.read()
        np_image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        aligned_faces = detect_and_align_faces(image)
        if aligned_faces:
            encodings = [face_recognition.face_encodings(face)[0] for face in aligned_faces if face_recognition.face_encodings(face)]
            return encodings if encodings else None
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Face detection and alignment
def detect_and_align_faces(image):
    resized_image = cv2.resize(image, (800, int(800 / (image.shape[1] / image.shape[0]))))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    aligned_faces = [dlib.get_face_chip(resized_image, shape_predictor(gray, face), size=256) for face in faces]
    return aligned_faces if aligned_faces else None

# Function to save multiple encodings for each person in Firebase
def add_person(name, image_files):
    all_encodings = []
    for image_file in image_files:
        encodings = load_and_encode(image_file)
        if encodings:
            all_encodings.extend(encodings)
    
    if all_encodings:
        average_encoding = np.mean(all_encodings, axis=0).tolist()
        ref.child(name).set({"encoding": average_encoding})
        st.success(f"Person {name} added successfully with {len(image_files)} images!")
    else:
        st.error("No face detected in the images.")

# Function to recognize person using multiple face embeddings
def recognize_face(image_file):
    unknown_encodings = load_and_encode(image_file)
    if not unknown_encodings:
        st.error("No face detected in the image.")
        return

    matches = set()
    for unknown_encoding in unknown_encodings:
        for name, data in ref.get().items():
            known_encoding = np.array(data['encoding'])
            if face_recognition.compare_faces([known_encoding], unknown_encoding)[0]:
                matches.add(name)

    if matches:
        st.success(f"Matched with: {', '.join(matches)}")
    else:
        st.error("No matches found.")

# Streamlit UI
st.title("Face Recognition App")

menu = ["Add Person", "Recognize Face"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Add Person":
    st.subheader("Add a New Person")
    name = st.text_input("Enter name:")
    image_files = st.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if st.button("Add Person") and image_files and name:
        add_person(name, image_files)

elif choice == "Recognize Face":
    st.subheader("Recognize Face")
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if st.button("Recognize") and image_file:
        recognize_face(image_file)
