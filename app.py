import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import face_recognition
import numpy as np
import cv2
import dlib
from moviepy.editor import VideoFileClip

# Firebase setup (initialize using your Firebase credentials)
current_directory = os.path.dirname(os.path.abspath(__file__))
firebase_cred_path = os.path.join(current_directory, "espiotproject1-firebase-adminsdk-856yr-1b9ae0c516.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_cred_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://espiotproject1-default-rtdb.firebaseio.com'
    })

# Reference to Firebase
ref = db.reference('/')

# Dlib model for face detection and landmarking
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)

# Function to load and encode an image
def load_and_encode(image):
    try:
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    aligned_faces = [dlib.get_face_chip(image, shape_predictor(gray, face), size=256) for face in faces]
    return aligned_faces if aligned_faces else None

# Function to process multiple images for a person
def add_person_multiple_images(name, image_files):
    all_encodings = []
    for image_file in image_files:
        image_bytes = image_file.read()
        np_image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        encodings = load_and_encode(image)
        if encodings:
            all_encodings.extend(encodings)
    
    if all_encodings:
        mean_encoding = np.mean(all_encodings, axis=0).tolist()
        ref.child(name).set({"encoding": mean_encoding})
        st.success(f"Person {name} added successfully with multiple images!")
    else:
        st.error("No faces detected in the images.")

# Function to extract frames from a video
def extract_frames_from_video(video_file, num_frames=10):
    video = VideoFileClip(video_file)
    duration = video.duration
    random_times = sorted(np.random.uniform(0, duration, num_frames))

    frames = []
    for time in random_times:
        frame = video.get_frame(time)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert to BGR for OpenCV

    return frames

# Add person via video input
def add_person_from_video(name, video_file):
    frames = extract_frames_from_video(video_file)
    all_encodings = []
    for frame in frames:
        encodings = load_and_encode(frame)
        if encodings:
            all_encodings.extend(encodings)
    
    if all_encodings:
        mean_encoding = np.mean(all_encodings, axis=0).tolist()
        ref.child(name).set({"encoding": mean_encoding})
        st.success(f"Person {name} added successfully with video input!")
    else:
        st.error("No faces detected in the video frames.")

# Recognize person from image
def recognize_face(image_file):
    image_bytes = image_file.read()
    np_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    unknown_encodings = load_and_encode(image)
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

menu = st.radio("Select an option", ["Add Person (Image Input)", "Add Person (Video Input)", "Recognize Face"])

if menu == "Add Person (Image Input)":
    st.subheader("Add a New Person with Multiple Images")
    name = st.text_input("Enter name:")
    image_files = st.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if st.button("Add Person") and image_files and name:
        add_person_multiple_images(name, image_files)

elif menu == "Add Person (Video Input)":
    st.subheader("Add a New Person with Video Input")
    name = st.text_input("Enter name:")
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if st.button("Add Person") and video_file and name:
        add_person_from_video(name, video_file)

elif menu == "Recognize Face":
    st.subheader("Recognize Face from an Image")
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if st.button("Recognize") and image_file:
        recognize_face(image_file)
