import cv2
import time
import math
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ======================================================
# 1. LOAD MODELS
# ======================================================
model = YOLO("yolov8n.pt")

base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_lite.task"
)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# ======================================================
# 2. VIDEO SETUP
# ======================================================
cap = cv2.VideoCapture(0)
w, h = int(cap.get(3)), int(cap.get(4))

out = cv2.VideoWriter(
    "child_safety_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
    (w, h)
)

# ======================================================
# 3. STATE VARIABLES
# ======================================================
prev_hip_y = None
prev_time = None
posture_buffer = deque(maxlen=15)
unsafe_start = None

print("High-Accuracy Child Safety System Running... Press Q to exit")

# ======================================================
# 4. HELPER FUNCTION
# ======================================================
def majority_vote(buffer):
    return max(set(buffer), key=buffer.count)

# ======================================================
# 5. MAIN LOOP
# ======================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.6)

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person = frame[y1:y2, x1:x2]
            if person.size == 0:
                continue

            # ------------------------------
            # Pose Estimation
            # ------------------------------
            rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(mp.ImageFormat.SRGB, rgb)
            pose = pose_landmarker.detect(mp_img)

            posture = "UNKNOWN"

            if pose.pose_landmarks:
                lm = pose.pose_landmarks[0]

                # ------------------------------
                # KEY LANDMARKS
                # ------------------------------
                nose = lm[0]
                ls, rs = lm[11], lm[12]
                lw, rw = lm[15], lm[16]
                lh, rh = lm[23], lm[24]
                lk, rk = lm[25], lm[26]

                head_y = nose.y
                hip_y = (lh.y + rh.y) / 2
                knee_y = (lk.y + rk.y) / 2

                body_height = abs(head_y - knee_y)

                shoulder_x = (ls.x + rs.x) / 2
                shoulder_y = (ls.y + rs.y) / 2
                hip_x = (lh.x + rh.x) / 2

                dx = shoulder_x - hip_x
                dy = shoulder_y - hip_y
                angle = abs(math.degrees(math.atan2(dy, dx)))

                # ------------------------------
                # FALL DETECTION (VELOCITY)
                # ------------------------------
                fall_detected = False
                now = time.time()

                if prev_hip_y is not None and prev_time is not None:
                    dy_v = hip_y - prev_hip_y
                    dt = now - prev_time
                    if dt > 0 and dy_v / dt > 1.1:
                        fall_detected = True

                prev_hip_y = hip_y
                prev_time = now

                # ------------------------------
                # POSTURE CLASSIFICATION
                # ------------------------------
                if fall_detected and body_height < 0.45:
                    posture = "FALLING"
                elif angle < 30 and body_height < 0.45:
                    posture = "LYING"
                elif body_height < 0.65:
                    posture = "SITTING"
                else:
                    posture = "STANDING"

                posture_buffer.append(posture)

                # ------------------------------
                # RED DOTS (BODY PART IDENTIFICATION)
                # ------------------------------
                body_parts = {
                    "Head": nose,
                    "L-Shoulder": ls,
                    "R-Shoulder": rs,
                    "L-Hand": lw,
                    "R-Hand": rw,
                    "L-Hip": lh,
                    "R-Hip": rh,
                    "L-Knee": lk,
                    "R-Knee": rk
                }

                for name, p in body_parts.items():
                    if p.visibility < 0.4:
                        continue

                    cx = int(x1 + p.x * (x2 - x1))
                    cy = int(y1 + p.y * (y2 - y1))q

                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(frame, name,
                                (cx + 4, cy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 255, 255),
                                1)

            # ------------------------------
            # FINAL POSTURE (SMOOTHED)
            # ------------------------------
            final_posture = majority_vote(posture_buffer) if posture_buffer else "Detecting"

            # ------------------------------
            # SAFETY DECISION
            # ------------------------------
            if final_posture in ["FALLING", "LYING"]:
                safety = "UNSAFE"
            else:
                safety = "SAFE"

            if safety == "UNSAFE":
                if unsafe_start is None:
                    unsafe_start = time.time()
                elif time.time() - unsafe_start > 3:
                    safety = "ALERT"
            else:
                unsafe_start = None

            # ------------------------------
            # DISPLAY
            # ------------------------------
            color = (0, 255, 0) if safety == "SAFE" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"Posture : {final_posture}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)

            cv2.putText(frame, f"Status  : {safety}",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color, 2)

    cv2.imshow("Child Safety Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================================================
# 6. CLEANUP
# ======================================================
cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved as child_safety_output.mp4")
