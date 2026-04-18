import cv2
import time
import requests
from collections import deque
import winsound
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# TELEGRAM SETTINGS
# -------------------------------
BOT_TOKEN = "Can't Expose Openly"
CHAT_ID = "6680941372"

ENABLE_TELEGRAM = True
last_telegram_time = 0
TELEGRAM_INTERVAL = 20

def send_telegram_alert(message):
    global last_telegram_time

    if not ENABLE_TELEGRAM:
        return

    if time.time() - last_telegram_time < TELEGRAM_INTERVAL:
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:
        requests.post(url, data=payload, timeout=5)
        print("📩 Telegram Alert Sent")
        last_telegram_time = time.time()
    except:
        print("Telegram Error")


# -------------------------------
# Load YOLO
# -------------------------------
model = YOLO("yolov8n.pt")

# -------------------------------
# MediaPipe Pose
# -------------------------------
base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_lite.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# -------------------------------
# Video Setup
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

out = cv2.VideoWriter(
    "child_safety_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
    (640, 480)
)

posture_buffer = deque(maxlen=5)
last_beep_time = 0
BEEP_INTERVAL = 1

print(" FAST Child Safety Detection Running... Press Q to stop")

# -------------------------------
# Posture Detection Logic
# -------------------------------
def detect_position(lm, bw, bh):

    shoulder_y = (lm[11].y + lm[12].y) / 2
    hip_y = (lm[23].y + lm[24].y) / 2
    ankle_y = (lm[27].y + lm[28].y) / 2

    body_height = abs(shoulder_y - ankle_y)
    aspect_ratio = bh / (bw + 1e-6)

    # Lying / Fall
    if aspect_ratio < 0.8 or body_height < 0.35:
        return "LYING/FALLEN"

    # Sitting
    if hip_y > shoulder_y and body_height < 0.6:
        return "SITTING"

    return "STANDING"


# -------------------------------
# Main Loop
# -------------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.6, verbose=False)

    final_pos = "Detecting"
    is_safe = True

    for r in results:
        for box in r.boxes:

            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            mp_img = mp.Image(
                mp.ImageFormat.SRGB,
                cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            )

            pose_result = pose_landmarker.detect(mp_img)

            if pose_result.pose_landmarks:

                lm = pose_result.pose_landmarks[0]
                posture = detect_position(lm, bw, bh)
                posture_buffer.append(posture)

                # Draw red keypoints with names
                keypoints = {
                    0: "Head",
                    11: "LShoulder",
                    12: "RShoulder",
                    15: "LHand",
                    16: "RHand"
                }

                for idx, name in keypoints.items():
                    pt = lm[idx]

                    cx = int(x1 + pt.x * bw)
                    cy = int(y1 + pt.y * bh)

                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                    cv2.putText(frame,
                                name,
                                (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1)

            # Bounding box (color updated later)
            box_coords = (x1, y1, x2, y2)

    if posture_buffer:
        final_pos = max(set(posture_buffer), key=list(posture_buffer).count)
        is_safe = final_pos in ["STANDING", "SITTING"]

    if not is_safe:
        if time.time() - last_beep_time > BEEP_INTERVAL:
            winsound.Beep(1500, 600)
            last_beep_time = time.time()

        send_telegram_alert("🚨 ALERT: Person Fallen or Lying Detected!")

    color = (0,255,0) if is_safe else (0,0,255)

    # Draw bounding box
    try:
        cv2.rectangle(frame,
                      (box_coords[0], box_coords[1]),
                      (box_coords[2], box_coords[3]),
                      color,
                      3)
    except:
        pass

    cv2.putText(frame,
                f"{final_pos} - {'SAFE' if is_safe else 'UNSAFE'}",
                (30,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3)

    cv2.imshow("Child Safety Final System", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video saved as child_safety_output.mp4")
