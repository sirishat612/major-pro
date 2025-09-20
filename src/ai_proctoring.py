import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Eye and Head Pose + Emotion Detection
class ProctorAI:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    # ----------------------------
    # 1. Eye Gaze Tracking
    # ----------------------------
    def detect_gaze(self, landmarks, frame_width):
        # Get left and right eye landmark indexes (example indexes from Mediapipe)
        left_eye = [33, 133]  # Left corner points
        right_eye = [362, 263]  # Right corner points
        iris_left = 468
        iris_right = 473

        left_ratio = (landmarks[iris_left].x - landmarks[left_eye[0]].x) / (landmarks[left_eye[1]].x - landmarks[left_eye[0]].x)
        right_ratio = (landmarks[iris_right].x - landmarks[right_eye[0]].x) / (landmarks[right_eye[1]].x - landmarks[right_eye[0]].x)

        gaze_direction = "Center"
        if left_ratio < 0.35 and right_ratio < 0.35:
            gaze_direction = "Looking Left"
        elif left_ratio > 0.65 and right_ratio > 0.65:
            gaze_direction = "Looking Right"

        return gaze_direction

    # ----------------------------
    # 2. Head Pose Estimation
    # ----------------------------
    def detect_head_pose(self, landmarks, frame_shape):
        h, w = frame_shape[:2]

        # Select important facial points (2D image points)
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),     # Nose tip
            (landmarks[199].x * w, landmarks[199].y * h), # Chin
            (landmarks[33].x * w, landmarks[33].y * h),   # Left eye corner
            (landmarks[263].x * w, landmarks[263].y * h), # Right eye corner
            (landmarks[61].x * w, landmarks[61].y * h),   # Left mouth corner
            (landmarks[291].x * w, landmarks[291].y * h)  # Right mouth corner
        ], dtype="double")

        # Approximate 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),    # Nose tip
            (0.0, -330.0, -65.0), # Chin
            (-225.0, 170.0, -135.0), # Left eye
            (225.0, 170.0, -135.0),  # Right eye
            (-150.0, -150.0, -125.0), # Left mouth
            (150.0, -150.0, -125.0)   # Right mouth
        ])

        # Camera internals
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4,1)) # No lens distortion
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        # Convert rotation to degrees
        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        pitch, yaw, roll = [float(angle) for angle in euler_angles]

        head_orientation = "Forward"
        if yaw > 30:
            head_orientation = "Looking Left"
        elif yaw < -30:
            head_orientation = "Looking Right"
        elif pitch > 20:
            head_orientation = "Looking Down"
        elif pitch < -15:
            head_orientation = "Looking Up"

        return head_orientation

    # ----------------------------
    # 3. Emotion Detection
    # ----------------------------
    def detect_emotion(self, frame, face_box):
        x, y, w, h = face_box
        face_crop = frame[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
        except:
            dominant_emotion = "Unknown"

        return dominant_emotion
