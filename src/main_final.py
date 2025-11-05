import os
import cv2
import yaml
import time
import pygame
import sys
from datetime import datetime

# --- Imports from your existing modules ---
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
from reporting.report_generator import ReportGenerator
from ai_proctoring import ProctorAI


# ---------- CONFIG ----------
def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


# ---------- DISPLAY ----------
def display_detection_results(frame, results, current_alert, unique_alert_count, total_alert_types):
    h, w = frame.shape[:2]

    # --- Header Bar ---
    cv2.rectangle(frame, (0, 0), (w, 60), (25, 25, 112), -1)
    cv2.putText(frame, "Enhanced Online Proctoring System", (20, 30),
                cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, results['timestamp'], (w - 220, 50),
                cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 2)

    # --- Status Line ---
    cv2.rectangle(frame, (0, 65), (w, 95), (230, 230, 230), -1)
    status = (
        f"Face: {'Present' if results['face_present'] else 'Absent'}   Gaze: {results['gaze_direction']}   "
        f"Eyes: {'Open' if results['eye_ratio'] > 0.25 else 'Closed'}   Mouth: {'Moving' if results['mouth_moving'] else 'Still'}"
    )
    cv2.putText(frame, status, (15, 85), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 2)

    # --- Alert Bar ---
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 215, 255), -1)
    if current_alert:
        alert_display = f"ALERT: {current_alert} | Unique Alerts: {unique_alert_count}/{total_alert_types}"
    else:
        alert_display = "Status: All clear"
    cv2.putText(frame, alert_display, (15, h - 12),
                cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 2)


# ---------- MAIN ----------
def main():
    # Load configuration
    config = load_config()

    # Initialize components
    alert_logger = AlertLogger(config)
    video_recorder = VideoRecorder(config)
    screen_recorder = ScreenRecorder(config)
    audio_monitor = AudioMonitor(config)
    alert_system = AlertSystem(config)  # Initialize alert system
    audio_monitor.alert_system = alert_system  # Connect alert system to audio monitor
    report_generator = ReportGenerator(config)

    # --- Initialize pygame for alert sounds ---
    pygame.mixer.init()
    def play_alert_sound():
        try:
            sound_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'alert.wav')
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()
        except Exception:
            pass

    # --- Audio Monitoring ---
    if config['detection']['audio_monitoring']:
        audio_monitor.start()

    # --- Detectors ---
    detectors = [
        FaceDetector(config),
        EyeTracker(config),
        MouthMonitor(config),
        MultiFaceDetector(config),
        ObjectDetector(config)
    ]

    # Set alert logger for all detectors
    for detector in detectors:
        if hasattr(detector, 'set_alert_logger'):
            detector.set_alert_logger(alert_logger)

    # --- Webcam Setup ---
    cap = cv2.VideoCapture(config['video'].get('source', 0), cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Webcam not accessible. Check permissions or try different source index.")
        return

    print("✅ Webcam initialized successfully.")
    print("👉 Press 'Q' to quit at any time.")

    # --- Set up full screen window ---
    cv2.namedWindow("Enhanced Online Proctoring System", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Enhanced Online Proctoring System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Force full screen mode
    cv2.resizeWindow("Enhanced Online Proctoring System", 1920, 1080)

    # --- Recording setup ---
    video_recorder.start_recording()
    if config['screen']['recording']:
        screen_recorder.start_recording()

    # --- Alert System Variables ---
    alert_types = {
        "Face disappear": False,
        "Mobile detected": False,
        "Don't speak, mouth movement": False,
        "Audio detected, don't talk": False,
        "Look straight": False,
        "Multiple faces detected": False
    }
    total_alert_types = len(alert_types)
    unique_alert_count = 0
    last_alert_time = 0
    cooldown = 2  # seconds between alerts
    active_alerts = {}  # Dictionary to track active alerts with timestamps
    alert_display_duration = 2  # seconds to display each alert

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Frame not captured from webcam.")
                break

            # --- Detection Results ---
            results = {
                'face_present': False,
                'gaze_direction': 'Center',
                'eye_ratio': 0.3,
                'mouth_moving': False,
                'multiple_faces': False,
                'objects_detected': False,
                'emotion': 'Neutral',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'audio_detected': audio_monitor.is_noise_detected() if hasattr(audio_monitor, 'is_noise_detected') else False
            }

            try:
                results['face_present'] = detectors[0].detect_face(frame)
                results['gaze_direction'], results['eye_ratio'] = detectors[1].track_eyes(frame)
                results['mouth_moving'] = detectors[2].monitor_mouth(frame)
                results['multiple_faces'] = detectors[3].detect_multiple_faces(frame)
                results['objects_detected'] = detectors[4].detect_objects(frame)
                if hasattr(audio_monitor, 'is_noise_detected') and audio_monitor.is_noise_detected():
                    results['audio_detected'] = True
            except Exception as e:
                print("Detection Error:", e)

            # --- Alert Conditions ---
            triggered_alerts = []
            if not results['face_present']:
                triggered_alerts.append("Face disappear")
            if results['objects_detected']:
                triggered_alerts.append("Mobile detected")
            if results['mouth_moving']:
                triggered_alerts.append("Don't speak, mouth movement")
            if results['audio_detected']:
                triggered_alerts.append("Audio detected, don't talk")
            if results['gaze_direction'] not in ['Center', 'center']:
                triggered_alerts.append("Look straight")
            if results['multiple_faces']:
                triggered_alerts.append("Multiple faces detected")

            # --- Update active alerts with timestamps ---
            now = time.time()
            for alert in triggered_alerts:
                active_alerts[alert] = now

            # --- Remove expired alerts ---
            expired_alerts = [alert for alert, timestamp in active_alerts.items()
                            if now - timestamp > alert_display_duration]
            for alert in expired_alerts:
                del active_alerts[alert]

            # --- Set current alerts for display (only active ones) ---
            current_alert = " | ".join(sorted(active_alerts.keys())) if active_alerts else ""

            # --- Update unique alerts ---
            if triggered_alerts and (now - last_alert_time > cooldown):
                for a in triggered_alerts:
                    if a in alert_types and not alert_types[a]:
                        alert_types[a] = True
                        play_alert_sound()
                        # Trigger voice alert for each new alert type
                        if a == "Face disappear":
                            alert_system.speak_alert("FACE_DISAPPEARED")
                        elif a == "Mobile detected":
                            alert_system.speak_alert("OBJECT_DETECTED")
                        elif a == "Don't speak, mouth movement":
                            alert_system.speak_alert("MOUTH_MOVING")
                        elif a == "Audio detected, don't talk":
                            alert_system.speak_alert("VOICE_DETECTED")
                        elif a == "Look straight":
                            alert_system.speak_alert("GAZE_AWAY")
                        elif a == "Multiple faces detected":
                            alert_system.speak_alert("MULTIPLE_FACES")
                        print(f"⚠ Alert Triggered: {a}")

                unique_alert_count = sum(1 for v in alert_types.values() if v)
                last_alert_time = now

            # --- Termination after all unique alerts ---
            if unique_alert_count == total_alert_types:
                cv2.rectangle(frame, (0, frame.shape[0] - 60),
                              (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                cv2.putText(frame, "Session Terminated - All Alerts Triggered 🚫",
                            (20, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("Enhanced Online Proctoring System", frame)
                cv2.waitKey(3000)
                print("Session terminated: All alert types triggered.")
                break

            # --- Display Results ---
            display_detection_results(frame, results, current_alert, unique_alert_count, total_alert_types)
            video_recorder.record_frame(frame)
            cv2.imshow("Enhanced Online Proctoring System", frame)

            # --- Quit manually ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🟡 Session terminated by user.")
                break

    finally:
        # Cleanup
        if config['screen']['recording']:
            screen_recorder.stop_recording()
        video_recorder.stop_recording()
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

        # --- Report Generation ---
        student_info = {
            'id': 'STUDENT_001',
            'name': 'John Doe',
            'exam': 'Final Examination',
            'course': 'Computer Science 101'
        }
        violations = [f"Alert {i+1}" for i in range(unique_alert_count)]
        report_path = report_generator.generate_report(student_info, violations)
        print(f"✅ Report generated: {report_path}")


if __name__ == "__main__":
    main()
