import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import face_recognition
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime
import dlib

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

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
    """Add face encoding to Firebase with PRN structure"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    ref = db.reference(f'FaceData/{current_date}/{prn}')
    
    data = {
        'name': name,
        'encoding': image_encoding.tolist(),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    ref.set(data)
    return True

def mark_attendance(prn, name, confidence):
    """Mark attendance in Firebase"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    attendance_ref = db.reference(f'Attendance/{current_date}/{prn}')
    
    data = {
        'name': name,
        'time': datetime.now().strftime("%H:%M:%S"),
        'confidence': float(confidence),
        'status': 'Present'
    }
    attendance_ref.set(data)

def get_known_faces():
    """Retrieve known faces from Firebase"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    ref = db.reference(f'FaceData/{current_date}')
    face_data = ref.get()
    
    known_encodings = []
    known_prns = []
    known_names = []
    
    if face_data:
        for prn, data in face_data.items():
            if 'encoding' in data:
                known_encodings.append(np.array(data['encoding']))
                known_prns.append(prn)
                known_names.append(data['name'])
    return known_encodings, known_prns, known_names

def detect_and_align_faces(image):
    """Detect and align faces using dlib"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    aligned_faces = []
    face_locations = []
    
    for face in faces:
        shape = shape_predictor(gray, face)
        aligned_face = dlib.get_face_chip(image, shape, size=256)
        aligned_faces.append(aligned_face)
        face_locations.append((face.top(), face.right(), face.bottom(), face.left()))
    
    return aligned_faces, face_locations

def process_image_for_recognition(image):
    """Process image and recognize faces"""
    aligned_faces, face_locations = detect_and_align_faces(image)
    known_encodings, known_prns, known_names = get_known_faces()
    
    verification_needed = []
    recognition_results = []
    
    for idx, aligned_face in enumerate(aligned_faces):
        face_encoding = face_recognition.face_encodings(aligned_face)[0]
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            confidence = (1 - face_distances[best_match_index]) * 100
            
            if confidence >= 50:
                prn = known_prns[best_match_index]
                name = known_names[best_match_index]
                mark_attendance(prn, name, confidence)
                recognition_results.append({
                    'location': face_locations[idx],
                    'name': name,
                    'prn': prn,
                    'confidence': confidence,
                    'status': 'Confirmed'
                })
            else:
                verification_needed.append({
                    'location': face_locations[idx],
                    'face_image': aligned_face,
                    'encoding': face_encoding,
                    'matched_prn': known_prns[best_match_index],
                    'matched_name': known_names[best_match_index],
                    'confidence': confidence
                })
    
    return recognition_results, verification_needed

def draw_results(image, recognition_results):
    """Draw recognition results on image"""
    for result in recognition_results:
        top, right, bottom, left = result['location']
        color = (36, 255, 12) if result['status'] == 'Confirmed' else (0, 0, 255)
        
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        text = f"{result['name']} ({result['confidence']:.1f}%)"
        cv2.putText(image, text, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return image

# Streamlit UI
st.title("Enhanced Face Recognition Attendance System")

# Sidebar for database management
st.sidebar.header("Database Management")
with st.sidebar.expander("Add New Student"):
    prn = st.text_input("Enter PRN")
    name = st.text_input("Enter Name")
    upload_for_db = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
    
    if st.button("Add to Database") and prn and name and upload_for_db:
        file_bytes = np.asarray(bytearray(upload_for_db.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        aligned_faces, _ = detect_and_align_faces(img)
        
        if aligned_faces:
            encoding = face_recognition.face_encodings(aligned_faces[0])[0]
            if add_to_firebase(prn, name, encoding):
                st.success(f"Added {name} (PRN: {prn}) to database!")
        else:
            st.error("No face detected in the image")

# Main interface
st.header("Take Attendance")
uploaded_file = st.file_uploader("Upload image for attendance", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and process image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    with st.spinner("Processing image..."):
        recognition_results, verification_needed = process_image_for_recognition(image)
        processed_image = draw_results(image.copy(), recognition_results)
    
    # Display results
    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), 
             caption="Processed Image", 
             use_column_width=True)
    
    # Handle verifications needed
    if verification_needed:
        st.subheader("Verification Needed")
        for idx, verify_data in enumerate(verification_needed):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(cv2.cvtColor(verify_data['face_image'], cv2.COLOR_BGR2RGB),
                        caption=f"Unconfirmed Face {idx+1}",
                        width=200)
            
            with col2:
                st.write(f"Possible match: {verify_data['matched_name']}")
                st.write(f"PRN: {verify_data['matched_prn']}")
                st.write(f"Confidence: {verify_data['confidence']:.1f}%")
                
                if st.button(f"Confirm {verify_data['matched_name']}", key=f"confirm_{idx}"):
                    mark_attendance(verify_data['matched_prn'], 
                                 verify_data['matched_name'],
                                 verify_data['confidence'])
                    st.success(f"Attendance marked for {verify_data['matched_name']}")
                
                if st.button("Not the same person", key=f"reject_{idx}"):
                    st.info("Rejected - No attendance marked")

# View Attendance Records
with st.sidebar.expander("View Attendance Records"):
    date_to_view = st.date_input("Select Date", datetime.now())
    formatted_date = date_to_view.strftime("%Y-%m-%d")
    
    attendance_ref = db.reference(f'Attendance/{formatted_date}')
    attendance_data = attendance_ref.get()
    
    if attendance_data:
        df = pd.DataFrame.from_dict(attendance_data, orient='index')
        st.dataframe(df)
        
        # Download attendance report
        csv = df.to_csv().encode('utf-8')
        st.download_button(
            "Download Attendance Report",
            csv,
            f"attendance_{formatted_date}.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.info("No attendance records for selected date")