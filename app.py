import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import face_recognition
import numpy as np
import cv2
import dlib
from moviepy.editor import VideoFileClip
import random

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
shape_predictor_path = "shape_predictor_68_face_landmarks.dat" #project2\shape_predictor_68_face_landmarks.dat
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

# Function to extract frames from video and process embeddings
def process_video_for_embeddings(video_file, name):
    video_path = os.path.join(current_directory, video_file.name)
    with open(video_path, 'wb') as f:
        f.write(video_file.getvalue())

    output_folder = os.path.join(current_directory, "output_frames")
    num_frames = 10  # Number of frames to extract
    extract_random_frames(video_path, output_folder, num_frames=num_frames)
    
    encodings = []
    for frame_file in os.listdir(output_folder):
        frame_path = os.path.join(output_folder, frame_file)
        with open(frame_path, 'rb') as img_file:
            img_encodings = load_and_encode(img_file)
            if img_encodings:
                encodings.extend(img_encodings)

    if encodings:
        average_encoding = np.mean(encodings, axis=0).tolist()
        ref.child(name).set({"encoding": average_encoding})
        st.success(f"Person {name} added successfully with video input!")
    else:
        st.error("No face detected in the video frames.")

# Function to extract random frames from a video
def extract_random_frames(video_path, output_folder, num_frames=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = VideoFileClip(video_path)
    duration = video.duration
    random_times = sorted(random.uniform(0, duration) for _ in range(num_frames))

    for i, time in enumerate(random_times):
        frame = video.get_frame(time)
        filename = f"frame_{i:04d}_{time:.2f}s.jpg"
        output_path = os.path.join(output_folder, filename)
        video.save_frame(output_path, t=time)

    video.close()

# Add person to database with image input
def add_person(name, image_file):
    encodings = load_and_encode(image_file)
    if encodings:
        encoding = encodings[0].tolist()
        ref.child(name).set({"encoding": encoding})
        st.success(f"Person {name} added successfully!")
    else:
        st.error("No face detected in the image.")

# Recognize person from image
def recognize_face(image_file):
    unknown_encodings = load_and_encode(image_file)
    if not unknown_encodings:
        st.error("No face detected in the image.")
        return

    matches = set()  # Use a set to store unique names
    for unknown_encoding in unknown_encodings:
        for name, data in ref.get().items():
            known_encoding = np.array(data['encoding'])
            if face_recognition.compare_faces([known_encoding], unknown_encoding)[0]:
                matches.add(name)  # Add the matched name to the set

    if matches:
        st.success(f"Matched with: {', '.join(matches)}")
    else:
        st.error("No matches found.")

# Streamlit UI
st.title("Face Recognition App")

menu = ["Add Person (Image Input)", "Add Person (Video Input)", "Recognize Face"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Add Person (Image Input)":
    st.subheader("Add a New Person (Using Image)")
    name = st.text_input("Enter name:")
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if st.button("Add Person") and image_file and name:
        add_person(name, image_file)

elif choice == "Add Person (Video Input)":
    st.subheader("Add a New Person (Using Video)")
    name = st.text_input("Enter name:")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if st.button("Add Person") and video_file and name:
        process_video_for_embeddings(video_file, name)

elif choice == "Recognize Face":
    st.subheader("Recognize Face")
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if st.button("Recognize") and image_file:
        recognize_face(image_file)
