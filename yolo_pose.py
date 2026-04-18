import cv2
import time
from collections import deque
import winsound
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------------------------
# Load Models
# -------------------------------------------------
model = YOLO("yolov8n.pt")

base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_lite.task"
)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# -------------------------------------------------
# Video Setup
# -------------------------------------------------
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 10

out = cv2.VideoWriter(
    "child_safety_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)

if not out.isOpened():
    print("ERROR: VideoWriter not opened")
    exit()

posture_buffer = deque(maxlen=10)

last_beep_time = 0
BEEP_INTERVAL = 1.0

print("Child Safety Detection running... Press Q to stop")

# -------------------------------------------------
# Posture Detection Logic
# -------------------------------------------------
def detect_position(landmarks, bw, bh):
    l_sh, r_sh = landmarks[11], landmarks[12]
    l_hip, r_hip = landmarks[23], landmarks[24]
    l_ank, r_ank = landmarks[27], landmarks[28]

    torso_height = abs(((l_sh.y + r_sh.y) / 2) - ((l_hip.y + r_hip.y) / 2))
    leg_height = abs(((l_hip.y + r_hip.y) / 2) - ((l_ank.y + r_ank.y) / 2))
    aspect_ratio = bh / (bw + 1e-6)

    if torso_height < 0.12 or aspect_ratio < 0.8:
        return "LYING/FALLEN"

    if leg_height < torso_height * 1.2 and aspect_ratio < 1.6:
        return "SITTING"

    if aspect_ratio >= 1.6:
        return "STANDING"

    return "SITTING"

# -------------------------------------------------
# Main Loop
# -------------------------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)

    final_pos = "Detecting"
    is_safe = True
    box_coords = None

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1
            box_coords = (x1, y1, x2, y2)

            # -------------------------------------------------
            # ✅ YOLO PERSON LABEL (ONLY ADDITION)
            # -------------------------------------------------
            cv2.putText(
                frame,
                "Person",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

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

                points_to_draw = [
                    (0, "Head"),
                    (11, "L-Shoulder"),
                    (12, "R-Shoulder"),
                    (15, "L-Hand"),
                    (16, "R-Hand"),
                    (23, "L-Hip"),
                    (24, "R-Hip")
                ]

                for idx, label in points_to_draw:
                    pt = lm[idx]
                    if hasattr(pt, "visibility") and pt.visibility < 0.6:
                        continue

                    cx = int(x1 + pt.x * bw)
                    cy = int(y1 + pt.y * bh)

                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        label,
                        (cx + 8, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )

    if posture_buffer:
        final_pos = max(set(posture_buffer), key=list(posture_buffer).count)
        is_safe = final_pos in ["STANDING", "SITTING"]

    current_time = time.time()
    if not is_safe:
        if current_time - last_beep_time >= BEEP_INTERVAL:
            winsound.Beep(1200, 700)
            last_beep_time = current_time
    else:
        last_beep_time = 0

    color = (0, 255, 0) if is_safe else (0, 0, 255)

    if box_coords:
        cv2.rectangle(
            frame,
            (box_coords[0], box_coords[1]),
            (box_coords[2], box_coords[3]),
            color,
            2
        )

    cv2.putText(
        frame,
        f"{final_pos} : {'SAFE' if is_safe else 'UNSAFE'}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

    cv2.imshow("Child Safety Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video saved as child_safety_output.mp4")
