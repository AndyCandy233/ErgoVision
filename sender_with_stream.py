import threading
import cv2
import math as m
import mediapipe as mp
import sys
import numpy as np
import time
import requests
import signal
import atexit
import datetime
from flask import Flask, Response

# --- Flask MJPEG Streaming Server ---
app = Flask(__name__)
global_frame = [None]  # Use list for mutability between threads

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            frame = global_frame[0]
            if frame is None:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask_server():
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)

# --- Posture Detection Functions ---
def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def findAngle(x1, y1, x2, y2):
    if y1 == 0:
        return 90
    numerator = (y2 - y1) * (-y1)
    denominator = (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1)
    value = numerator / denominator if denominator != 0 else 0
    value = max(min(value, 1), -1)
    theta = m.acos(value)
    return theta * (180 / m.pi)

def sound_alert():
    try:
        import winsound
        winsound.Beep(1000, 500)
    except ImportError:
        print('\a')

def sendWarning():
    print("Warning: Bad posture detected for too long!")
    sound_alert()

def sendInattentiveWarning():
    print("Warning: Inattentive detected!")
    sound_alert()

# --- Eye/Fatigue Detection Functions ---
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
LEFT_IRIS_IDX = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]
MOUTH_IDX = [13, 14, 78, 308]

FACE_LEFT_IDX = 234
FACE_RIGHT_IDX = 454

YAWN_THRESHOLD = 0.60
YAWN_CONSEC_FRAMES = 10

EYE_TO_FACE_RATIO_THRESHOLD = 0.45
LOOKING_UP_THRESHOLD = 0.65
LOOKING_DOWN_THRESHOLD = 1.35

# Debounce settings
ATTENTIVE_DEBOUNCE_FRAMES = 20
INATTENTIVE_DEBOUNCE_FRAMES = 20

def get_vertical_gaze_ratio(eye, iris_center):
    top = eye[1]
    bottom = eye[5]
    eye_height = np.linalg.norm(top - bottom)
    iris_to_top = np.linalg.norm(iris_center - top)
    ratio = iris_to_top / (eye_height + 1e-6)
    return ratio

def get_iris_center(iris):
    x = np.mean(iris[:, 0])
    y = np.mean(iris[:, 1])
    return np.array([x, y])

def get_gaze_direction(eye, iris_center):
    left_corner = eye[0]
    right_corner = eye[3]
    ratio = (iris_center[0] - left_corner[0]) / (right_corner[0] - left_corner[0] + 1e-6)
    if ratio < 0.35:
        return 'Left'
    elif ratio > 0.65:
        return 'Right'
    else:
        return 'Center'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def draw_fancy_box(img, pt1, pt2, color, alpha=0.8, r=20, thickness=2, fill=True):
    overlay = img.copy()
    if fill:
        cv2.rectangle(overlay, pt1, pt2, color, -1, cv2.LINE_AA)
    if r > 0:
        for dx in [0, 1]:
            for dy in [0, 1]:
                center = (
                    pt1[0] + r if dx == 0 else pt2[0] - r,
                    pt1[1] + r if dy == 0 else pt2[1] - r,
                )
                cv2.ellipse(overlay, center, (r, r), 0, 180 * dx, 180 + 180 * dx, color, -1 if fill else thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, pt1, pt2, color, thickness, cv2.LINE_AA)

USERNAME = "AndyCandy233"
REPORT_HOST = "http://10.37.123.168:8080/report"
REPORT_INTERVAL_SEC = 5

def get_current_utc():
    return datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

session_stats = {
    "user": USERNAME,
    "session_start": get_current_utc(),
    "bad_posture_time": 0.0,
    "bad_posture_events": 0,
    "inattentive_time": 0.0,
    "inattentive_events": 0,
    "yawn_count": 0,
    "frames": 0,
    "last_reported": get_current_utc(),
    "total_paused_time": 0.0
}
def send_report(final=False):
    if final:
        session_stats["session_end"] = get_current_utc()
    else:
        session_stats["last_reported"] = get_current_utc()
    try:
        payload = dict(session_stats)
        payload["final"] = final
        requests.post(REPORT_HOST, json=payload, timeout=2)
        print(f"Sent {'final' if final else 'periodic'} report to teacher.")
    except Exception as e:
        print(f"Failed to send report: {e}")

def handle_exit(*args):
    send_report(final=True)
    print("Session report sent. Exiting...")

atexit.register(handle_exit)
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

if __name__ == "__main__":
    # Start the Flask MJPEG streaming server in a thread
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    BAD_POSTURE_ALERT_SECONDS = 10
    SHOULDER_HEAD_DIST_THRESHOLD = 190

    INATTENTIVE_ALERT_SECONDS = 10

    alert_on_screen = False
    alert_counter = 0

    inattentive_alert_on_screen = False
    inattentive_alert_counter = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    timer_scale = 0.7
    timer_thick = 2
    alert_scale = 1.2
    alert_thick = 2

    attentive_frames = 0
    yawn_count = 0
    yawn_frame_counter = 0

    paused = False
    pause_start = None

    last_report = time.time()
    bad_frames = 0

    inattentive_time = 0.0
    inattentive_frames = 0

    pending_inattentive_frames = 0
    pending_attentive_frames = 0
    attention_state = "Attentive"

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            h, w = image.shape[:2]
            display_img = image.copy()

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            keypoints = pose.process(image_rgb)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            has_pose = keypoints.pose_landmarks is not None
            has_face = results.multi_face_landmarks is not None

            now = time.time()
            session_stats["frames"] += 1

            if not has_pose and not has_face:
                if not paused:
                    paused = True
                    pause_start = now
                paused_text = "PAUSED - No person detected"
                (txt_w, txt_h), baseline = cv2.getTextSize(paused_text, font, 1.0, 3)
                center = (int(w/2), int(h/2))
                rect_top_left = (center[0] - txt_w//2 - 32, center[1] - txt_h//2 - 32)
                rect_bottom_right = (center[0] + txt_w//2 + 32, center[1] + txt_h//2 + 32)
                draw_fancy_box(display_img, rect_top_left, rect_bottom_right, (30,30,30), alpha=0.82, r=32, thickness=0)
                cv2.putText(display_img, paused_text, (center[0]-txt_w//2, center[1]+txt_h//2-8),
                            font, 1.0, (200,200,200), 3, lineType=cv2.LINE_AA)
                footer_h = 38
                footer_y1 = h - footer_h
                footer_y2 = h
                draw_fancy_box(display_img, (0, footer_y1), (w, footer_y2), (25, 40, 70), alpha=0.85, r=0, thickness=0)
                cv2.putText(display_img, "Posture & Eye Fatigue Monitor - Press ESC to Quit", (20, h-13),
                            font, 0.62, (255,255,255), 1, lineType=cv2.LINE_AA)
                global_frame[0] = display_img.copy()
                cv2.imshow("Posture & Eye Fatigue Monitor", display_img)
                key = cv2.waitKey(300)
                if key & 0xFF == 27:
                    break
                continue
            else:
                if paused and pause_start:
                    session_stats["total_paused_time"] += now - pause_start
                paused = False
                pause_start = None

            bad_time = 0
            if keypoints.pose_landmarks:
                lm = keypoints.pose_landmarks
                lmPose = mp_pose.PoseLandmark

                try:
                    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
                    l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
                    l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
                    nose_x = int(lm.landmark[lmPose.NOSE].x * w)
                    nose_y = int(lm.landmark[lmPose.NOSE].y * h)

                    left_shoulder_head_dist = findDistance(l_shldr_x, l_shldr_y, nose_x, nose_y)
                    right_shoulder_head_dist = findDistance(r_shldr_x, r_shldr_y, nose_x, nose_y)

                    shoulders_too_close_to_head = (
                        left_shoulder_head_dist < SHOULDER_HEAD_DIST_THRESHOLD or
                        right_shoulder_head_dist < SHOULDER_HEAD_DIST_THRESHOLD
                    )

                    neck_inclination = findAngle(l_shldr_x, l_shldr_y, nose_x, nose_y)
                    torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

                    if (neck_inclination < 40 and torso_inclination < 10) and not shoulders_too_close_to_head:
                        bad_frames = 0
                    else:
                        bad_frames += 1

                    bad_time = (1 / fps) * bad_frames

                    box_color = (30, 30, 40)
                    text_color = (70, 255, 70) if bad_time == 0 else (40, 60, 255)
                    draw_fancy_box(display_img, (10, 10), (270, 60), box_color, alpha=0.7, r=18, thickness=2)
                    cv2.putText(display_img, f'Bad Posture: {round(bad_time, 1)}s', (28, 45), font, timer_scale, text_color, timer_thick)

                    alert_text = "BAD POSTURE ALERT!"
                    (alert_text_width, alert_text_height), baseline = cv2.getTextSize(alert_text, font, alert_scale, alert_thick)
                    alert_center = (int(w/2), int(h*0.28))
                    alert_rect_top_left = (alert_center[0] - alert_text_width//2 - 30, alert_center[1] - alert_text_height//2 - 22)
                    alert_rect_bottom_right = (alert_center[0] + alert_text_width//2 + 30, alert_center[1] + alert_text_height//2 + 22)

                    if bad_time > BAD_POSTURE_ALERT_SECONDS:
                        sendWarning()
                        alert_on_screen = True
                        alert_counter = 30
                        session_stats["bad_posture_events"] += 1
                        session_stats["bad_posture_time"] += bad_time
                        bad_frames = 0

                    if alert_on_screen:
                        draw_fancy_box(display_img, alert_rect_top_left, alert_rect_bottom_right, (0,0,0), alpha=0.75, r=28, thickness=0)
                        cv2.putText(
                            display_img, alert_text, 
                            (alert_center[0] - alert_text_width//2, alert_center[1] + alert_text_height//2 - 8),
                            font, alert_scale, (40, 60, 255), alert_thick, lineType=cv2.LINE_AA
                        )
                        alert_counter -= 1
                        if alert_counter < 0:
                            alert_on_screen = False

                    cv2.line(display_img, (l_shldr_x, l_shldr_y), (nose_x, nose_y), (255,180,80), 2)
                    cv2.line(display_img, (r_shldr_x, r_shldr_y), (nose_x, nose_y), (255,180,80), 2)
                    cv2.line(display_img, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), (100,200,255), 2)
                    cv2.line(display_img, (l_hip_x, l_hip_y), (r_shldr_x, r_shldr_y), (100,200,255), 2)
                    cv2.circle(display_img, (l_shldr_x, l_shldr_y), 5, (0,220,255), -1)
                    cv2.circle(display_img, (r_shldr_x, r_shldr_y), 5, (255,80,255), -1)
                    cv2.circle(display_img, (nose_x, nose_y), 6, (80,255,120), -1)
                    cv2.circle(display_img, (l_hip_x, l_hip_y), 5, (255,210,80), -1
                    )

                except Exception as e:
                    print(f"Error (pose): {e}")

            inattentive_this_frame = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = np.array(
                        [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
                    )
                    left_eye = landmarks[LEFT_EYE_IDX]
                    right_eye = landmarks[RIGHT_EYE_IDX]
                    left_iris = landmarks[LEFT_IRIS_IDX]
                    right_iris = landmarks[RIGHT_IRIS_IDX]
                    mouth = landmarks[MOUTH_IDX]

                    for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX + LEFT_IRIS_IDX + RIGHT_IRIS_IDX + MOUTH_IDX:
                        cv2.circle(display_img, tuple(landmarks[idx].astype(int)), 2, (0,255,0), -1)

                    left_iris_center = get_iris_center(left_iris)
                    right_iris_center = get_iris_center(right_iris)
                    left_gaze = get_gaze_direction(left_eye, left_iris_center)
                    right_gaze = get_gaze_direction(right_eye, right_iris_center)
                    
                    face_left = landmarks[FACE_LEFT_IDX]
                    face_right = landmarks[FACE_RIGHT_IDX]
                    face_width = np.linalg.norm(face_left - face_right)
                    eye_center_dist = np.linalg.norm(left_iris_center - right_iris_center)
                    eye_to_face_ratio = eye_center_dist / (face_width + 1e-6)

                    # --- Vertical gaze detection ---
                    left_vertical_ratio = get_vertical_gaze_ratio(left_eye, left_iris_center)
                    right_vertical_ratio = get_vertical_gaze_ratio(right_eye, right_iris_center)
                    looking_up = (left_vertical_ratio < LOOKING_UP_THRESHOLD and right_vertical_ratio < LOOKING_UP_THRESHOLD)
                    looking_down = (left_vertical_ratio > LOOKING_DOWN_THRESHOLD and right_vertical_ratio > LOOKING_DOWN_THRESHOLD)

                    # Combined inattentive logic: not center, offscreen, or up/down
                    if (left_gaze != "Center" or right_gaze != "Center") or (eye_to_face_ratio < EYE_TO_FACE_RATIO_THRESHOLD) or looking_up or looking_down:
                        inattentive_this_frame = True
                    else:
                        inattentive_this_frame = False

                    # Yawn detection
                    vertical = np.linalg.norm(mouth[0] - mouth[1])
                    horizontal = np.linalg.norm(mouth[2] - mouth[3])
                    mar = vertical / (horizontal + 1e-6)
                    if mar > YAWN_THRESHOLD:
                        yawn_frame_counter += 1
                    else:
                        if yawn_frame_counter >= YAWN_CONSEC_FRAMES:
                            yawn_count += 1
                            session_stats["yawn_count"] += 1
                        yawn_frame_counter = 0

                    # --- DEBOUNCE LOGIC ---
                    if inattentive_this_frame:
                        pending_inattentive_frames += 1
                        pending_attentive_frames = 0
                        if attention_state == "Attentive" and pending_inattentive_frames >= INATTENTIVE_DEBOUNCE_FRAMES:
                            attention_state = "Inattentive"
                            inattentive_frames = 1
                    else:
                        pending_attentive_frames += 1
                        pending_inattentive_frames = 0
                        if attention_state == "Inattentive" and pending_attentive_frames >= ATTENTIVE_DEBOUNCE_FRAMES:
                            attention_state = "Attentive"
                            inattentive_frames = 0

                    # UI: Draw fatigue stats in a rounded box at top right
                    box_x1 = w - 260
                    box_x2 = w - 15
                    box_y1 = 10
                    box_y2 = 90
                    draw_fancy_box(display_img, (box_x1, box_y1), (box_x2, box_y2), (30, 30, 40), alpha=0.7, r=18, thickness=2)
                    cv2.putText(display_img, f"Gaze: {attention_state}", (box_x1+18, box_y1+35), font, timer_scale, (255,255,0), timer_thick)
                    cv2.putText(display_img, f"Yawns: {yawn_count}", (box_x1+18, box_y1+65), font, timer_scale, (0,200,255), timer_thick)

            # Add inattentive time for this frame if inattentive
            if attention_state == "Inattentive":
                inattentive_time += 1.0 / fps
                session_stats["inattentive_time"] = inattentive_time
                inattentive_frames += 1
                if inattentive_time > 0 and inattentive_frames / fps >= INATTENTIVE_ALERT_SECONDS:
                    if not inattentive_alert_on_screen:
                        sendInattentiveWarning()
                        inattentive_alert_on_screen = True
                        inattentive_alert_counter = 30
                        session_stats["inattentive_events"] += 1
            else:
                inattentive_frames = 0

            if inattentive_alert_on_screen:
                alert_text = "INATTENTIVE ALERT!"
                (alert_text_width, alert_text_height), baseline = cv2.getTextSize(alert_text, font, alert_scale, alert_thick)
                alert_center = (int(w/2), int(h*0.45))
                alert_rect_top_left = (alert_center[0] - alert_text_width//2 - 30, alert_center[1] - alert_text_height//2 - 22)
                alert_rect_bottom_right = (alert_center[0] + alert_text_width//2 + 30, alert_center[1] + alert_text_height//2 + 22)
                draw_fancy_box(display_img, alert_rect_top_left, alert_rect_bottom_right, (0,0,0), alpha=0.75, r=28, thickness=0)
                cv2.putText(
                    display_img, alert_text, 
                    (alert_center[0] - alert_text_width//2, alert_center[1] + alert_text_height//2 - 8),
                    font, alert_scale, (0, 140, 255), alert_thick, lineType=cv2.LINE_AA
                )
                inattentive_alert_counter -= 1
                if inattentive_alert_counter < 0:
                    inattentive_alert_on_screen = False

            footer_h = 38
            footer_y1 = h - footer_h
            footer_y2 = h
            draw_fancy_box(display_img, (0, footer_y1), (w, footer_y2), (25, 40, 70), alpha=0.85, r=0, thickness=0)
            cv2.putText(display_img, "Posture & Eye Fatigue Monitor - Press ESC to Quit", (20, h-13),
                        font, 0.62, (255,255,255), 1, lineType=cv2.LINE_AA)

            # --- MJPEG Streaming update ---
            global_frame[0] = display_img.copy()

            if now - last_report > REPORT_INTERVAL_SEC:
                send_report(final=False)
                last_report = now

            cv2.imshow("Posture & Eye Fatigue Monitor", display_img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            if cv2.getWindowProperty('Posture & Eye Fatigue Monitor', cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()