import cv2
import yaml
from datetime import datetime
from detection.face_detection import FaceDetector
from detection.eye_tracking import EyeTracker
from detection.mouth_detection import MouthMonitor
from detection.object_detection import ObjectDetector
from detection.multi_face import MultiFaceDetector
from detection.audio_detection import AudioMonitor
from utils.video_utils import VideoRecorder
from utils.screen_capture import ScreenRecorder
from utils.logging import AlertLogger
from utils.alert_system import AlertSystem
from utils.violation_logger import ViolationLogger
from utils.screenshot_utils import ViolationCapturer
from reporting.report_generator import ReportGenerator
from ai_proctoring import ProctorAI


def load_config():
    with open('config/config.yaml') as f:
        return yaml.safe_load(f)

def display_detection_results(frame, results):
    h, w = frame.shape[:2]

    # Title Bar at the top
    cv2.rectangle(frame, (0, 0), (w, 60), (128, 0, 0), -1)
    cv2.putText(frame, "Enhanced Online Proctoring System", (20, 35),
               cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    # Left side status indicators (black color, removed "Head Not Forward")
    y_offset = 80
    line_height = 35

    # Status indicators in black
    status_items = [
        f"Face: {'Present' if results['face_present'] else 'Absent'}",
        f"Gaze: {results['gaze_direction']}",
        f"Eyes: {'Open' if results['eye_ratio'] > 0.25 else 'Closed'}",
        f"Mouth: {'Moving' if results['mouth_moving'] else 'Still'}",
        f"Emotion: {results['emotion']}"
    ]

    # Display status in black
    for item in status_items:
        cv2.putText(frame, item, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y_offset += line_height

    # Bottom black strip for triggered alerts only
    strip_height = 50
    cv2.rectangle(frame, (0, h - strip_height), (w, h), (0, 0, 0), -1)

    # Prepare triggered alerts only
    triggered_alerts = []
    alert_colors = {
        "Face Disappeared": (0, 0, 255),  # Red
        "Don't Speak": (0, 0, 255),       # Red
        "Mouth Movement": (0, 0, 255),    # Red
        "Mobile Detected": (0, 0, 255),   # Red
        "Look Straight": (0, 255, 255),   # Yellow
        "Multiple Faces": (0, 255, 255),  # Yellow
    }

    # Check for violations and warnings (triggered alerts)
    if not results['face_present']:
        triggered_alerts.append("Face Disappeared")
    if results['mouth_moving']:
        triggered_alerts.append("Don't Speak")
        triggered_alerts.append("Mouth Movement")
    if results['objects_detected']:
        triggered_alerts.append("Mobile Detected")
    if results['gaze_direction'] != "Center":
        triggered_alerts.append("Look Straight")
    if results['multiple_faces']:
        triggered_alerts.append("Multiple Faces")

    # Display triggered alerts in bottom strip (single line only)
    if triggered_alerts:
        # Main alert text - single line only
        alert_text = " | ".join(triggered_alerts)
        cv2.putText(frame, f"ALERTS: {alert_text}", (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        # No alerts - show "All Clear"
        cv2.putText(frame, "STATUS: All Clear", (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Timestamp in top right
    cv2.putText(frame, results['timestamp'],
               (frame.shape[1] - 250, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def main():
    config = load_config()
    alert_logger = AlertLogger(config)
    alert_system = AlertSystem(config)
    violation_capturer = ViolationCapturer(config)
    violation_logger = ViolationLogger(config)
    report_generator = ReportGenerator(config)

    # Initialize ProctorAI for advanced features
    proctor_ai = ProctorAI()

    student_info = {
        'id': 'STUDENT_001',
        'name': 'John Doe',
        'exam': 'Final Examination',
        'course': 'Computer Science 101'
    }

    # Initialize recorders
    video_recorder = VideoRecorder(config)
    screen_recorder = ScreenRecorder(config)

    # Initialize audio monitor
    audio_monitor = AudioMonitor(config)
    audio_monitor.alert_system = alert_system
    audio_monitor.alert_logger = alert_logger

    if config['detection']['audio_monitoring']:
        audio_monitor.start()

    try:
        if config['screen']['recording']:
            screen_recorder.start_recording()
        # Initialize detectors
        detectors = [
            FaceDetector(config),
            EyeTracker(config),
            MouthMonitor(config),
            MultiFaceDetector(config),
            ObjectDetector(config),
        ]

        for detector in detectors:
            if hasattr(detector, 'set_alert_logger'):
                detector.set_alert_logger(alert_logger)

        # Start webcam recording
        video_recorder.start_recording()
        cap = cv2.VideoCapture(config['video']['source'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['video']['resolution'][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['video']['resolution'][1])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = {
                'face_present': False,
                'gaze_direction': 'Center',
                'eye_ratio': 0.3,
                'mouth_moving': False,
                'multiple_faces': False,
                'objects_detected': False,
                'head_pose': 'Forward',
                'emotion': 'Neutral',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Perform detections
            results['face_present'] = detectors[0].detect_face(frame)
            results['gaze_direction'], results['eye_ratio'] = detectors[1].track_eyes(frame)
            results['mouth_moving'] = detectors[2].monitor_mouth(frame)
            results['multiple_faces'] = detectors[3].detect_multiple_faces(frame)
            results['objects_detected'] = detectors[4].detect_objects(frame)

            # Advanced AI features using ProctorAI
            if results['face_present']:
                # Get face landmarks for advanced detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = proctor_ai.face_mesh.process(rgb_frame)

                if face_results.multi_face_landmarks:
                    landmarks = face_results.multi_face_landmarks[0].landmark

                    # Eye gaze detection
                    results['gaze_direction'] = proctor_ai.detect_gaze(landmarks, frame.shape[1])

                    # Head pose estimation
                    results['head_pose'] = proctor_ai.detect_head_pose(landmarks, frame.shape)

                    # Get face bounding box for emotion detection
                    h, w = frame.shape[:2]
                    x_min = int(min([landmark.x for landmark in landmarks]) * w)
                    x_max = int(max([landmark.x for landmark in landmarks]) * w)
                    y_min = int(min([landmark.y for landmark in landmarks]) * h)
                    y_max = int(max([landmark.y for landmark in landmarks]) * h)

                    face_box = (x_min, y_min, x_max - x_min, y_max - y_min)
                    results['emotion'] = proctor_ai.detect_emotion(frame, face_box)

            # Violation detection and logging
            violations_detected = []

            if not results['face_present']:
                violation_type = "FACE_DISAPPEARED"
                alert_system.speak_alert(violation_type)
                violations_detected.append(violation_type)

            elif results['multiple_faces']:
                violation_type = "MULTIPLE_FACES"
                alert_system.speak_alert(violation_type)
                violations_detected.append(violation_type)

            elif results['objects_detected']:
                violation_type = "OBJECT_DETECTED"
                alert_system.speak_alert(violation_type)
                violations_detected.append(violation_type)

            elif results['gaze_direction'] != "Center":
                violation_type = "GAZE_AWAY"
                alert_system.speak_alert(violation_type)
                violations_detected.append(violation_type)

            elif results['head_pose'] != "Forward":
                violation_type = "HEAD_TURNED"
                alert_system.speak_alert(violation_type)
                violations_detected.append(violation_type)

            elif results['mouth_moving']:
                violation_type = "MOUTH_MOVING"
                alert_system.speak_alert(violation_type)
                violations_detected.append(violation_type)

            # Log all detected violations
            for violation_type in violations_detected:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                violation_image = violation_capturer.capture_violation(frame, violation_type, timestamp)
                violation_logger.log_violation(
                    violation_type,
                    timestamp,
                    {
                        'duration': '5+ seconds',
                        'frame': results,
                        'gaze_direction': results['gaze_direction'],
                        'head_pose': results['head_pose'],
                        'emotion': results['emotion']
                    }
                )

            # Display and record
            display_detection_results(frame, results)
            video_recorder.record_frame(frame)

            # Show preview
            cv2.imshow('Enhanced Online Proctoring System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        violations = violation_logger.get_violations()
        report_path = report_generator.generate_report(student_info, violations)
        print(f"Report generated: {report_path}")
        if config['screen']['recording']:
            screen_data = screen_recorder.stop_recording()
            print(f"Screen recording saved: {screen_data['filename']}")

        video_data = video_recorder.stop_recording()
        print(f"Webcam recording saved: {video_data['filename']}")

        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
