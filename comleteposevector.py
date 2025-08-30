import cv2
import mediapipe as mp
import numpy as np
import json
import math
from collections import OrderedDict

# ---------- CONFIG ----------
VIDEO_IN = "drunkenmonkeystyle.mp4"   # or 0 for webcam
OUTPUT_VIDEO = "output/annotated_output.mp4"
OUTPUT_JSON = "output/motion_log.json"
FOURCC = "mp4v"
PIXEL_TO_METER = 0.002
LEG_MASS = 10.0

# visual params
LEG_NORMAL_COLOR = (0, 255, 255)   # yellow
GRF_COLOR = (255, 0, 0)            # blue
PUNCH_COLOR = (0, 0, 255)          # red
ARC_COLOR = (0, 255, 0)            # green arcs
ANGLE_TEXT_COLOR = (255, 200, 0)   # yellow-ish
LANDMARK_COLOR = (255, 255, 255)   # white
EDGE_COLOR = (200, 200, 200)       # grey skeleton edges
FPS_FALLBACK = 25

# GRF clamp
GRF_MIN = -0.6
GRF_MAX = 0.6
GRF_SCALE_PX = 50

# --------------------------------

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

def clamp(x, a, b):
    return max(a, min(b, x))

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return None
    cosang = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def draw_half_arc_and_label(img, A, B, C, color=ARC_COLOR, radius_px=30):
    v1, v2 = A - B, C - B
    if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6: return
    ang1 = math.degrees(math.atan2(v1[1], v1[0])) % 360
    ang2 = math.degrees(math.atan2(v2[1], v2[0])) % 360
    diff = (ang2 - ang1 + 360) % 360
    start_ang, end_ang = (ang2, ang1) if diff > 180 else (ang1, ang2)
    center = (int(B[0]), int(B[1]))
    cv2.ellipse(img, center, (radius_px, radius_px), 0.0, start_ang, end_ang, color, 2)
    included = abs((math.degrees(math.atan2(v2[1], v2[0])) -
                    math.degrees(math.atan2(v1[1], v1[0])) + 360) % 360)
    if included > 180: included = 360 - included
    label_pos = (center[0] + 8, center[1] - radius_px - 6)
    cv2.putText(img, f"{included:.0f}°", label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, ANGLE_TEXT_COLOR, 1, cv2.LINE_AA)

def draw_arrow_with_label(img, origin, vec_px, color=(255,0,0), label=None, thickness=2):
    origin_pt = (int(origin[0]), int(origin[1]))
    end_pt = (int(origin[0] + vec_px[0]), int(origin[1] + vec_px[1]))
    cv2.arrowedLine(img, origin_pt, end_pt, color, thickness, tipLength=0.25)
    if label:
        perp = np.array([-vec_px[1], vec_px[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-6)
        label_pos = (int(end_pt[0] + perp[0]*8), int(end_pt[1] + perp[1]*8))
        cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    ANGLE_TEXT_COLOR, 1, cv2.LINE_AA)

# Open video
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise SystemExit(f"Cannot open input {VIDEO_IN}")

fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*FOURCC)
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

frame_idx, prev_ankles, log_records = 0, {"left": None, "right": None}, []

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(rgb)
        results_hands = hands.process(rgb)
        annotated = frame.copy()

        record = OrderedDict([
            ("frame", frame_idx), ("time_s", frame_idx / fps),
            ("keypoints", {}), ("forces", {}), ("joint_angles", {})
        ])

        # Pose landmarks
        if results_pose.pose_landmarks:
            lm = results_pose.pose_landmarks.landmark
            def L(i): return np.array([lm[i].x * w, lm[i].y * h])

            # BODY POINTS
            nose = L(mp_pose.PoseLandmark.NOSE.value)
            left_eye, right_eye = L(mp_pose.PoseLandmark.LEFT_EYE.value), L(mp_pose.PoseLandmark.RIGHT_EYE.value)
            left_ear, right_ear = L(mp_pose.PoseLandmark.LEFT_EAR.value), L(mp_pose.PoseLandmark.RIGHT_EAR.value)
            left_shoulder, right_shoulder = L(mp_pose.PoseLandmark.LEFT_SHOULDER.value), L(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            left_elbow, right_elbow = L(mp_pose.PoseLandmark.LEFT_ELBOW.value), L(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            left_wrist, right_wrist = L(mp_pose.PoseLandmark.LEFT_WRIST.value), L(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            left_hip, right_hip = L(mp_pose.PoseLandmark.LEFT_HIP.value), L(mp_pose.PoseLandmark.RIGHT_HIP.value)
            left_knee, right_knee = L(mp_pose.PoseLandmark.LEFT_KNEE.value), L(mp_pose.PoseLandmark.RIGHT_KNEE.value)
            left_ankle, right_ankle = L(mp_pose.PoseLandmark.LEFT_ANKLE.value), L(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

            # draw landmarks
            for p in [nose, left_eye, right_eye, left_ear, right_ear,
                      left_shoulder, right_shoulder, left_elbow, right_elbow,
                      left_wrist, right_wrist, left_hip, right_hip,
                      left_knee, right_knee, left_ankle, right_ankle]:
                cv2.circle(annotated, (int(p[0]), int(p[1])), 3, LANDMARK_COLOR, -1)

            # skeleton edges (body + head)
            edges = [
                (left_shoulder, right_shoulder),
                (left_shoulder, left_elbow), (left_elbow, left_wrist),
                (right_shoulder, right_elbow), (right_elbow, right_wrist),
                (left_shoulder, left_hip), (right_shoulder, right_hip),
                (left_hip, right_hip),
                (left_hip, left_knee), (left_knee, left_ankle),
                (right_hip, right_knee), (right_knee, right_ankle),
                (nose, left_eye), (nose, right_eye), (left_eye, left_ear), (right_eye, right_ear),
                (nose, left_shoulder), (nose, right_shoulder)
            ]
            for a, b in edges:
                cv2.line(annotated, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), EDGE_COLOR, 2)

            # --- Joint half-arcs (unchanged) ---
            for A, B, C in [
                (left_shoulder, left_elbow, left_wrist),
                (right_shoulder, right_elbow, right_wrist),
                (left_hip, left_knee, left_ankle),
                (right_hip, right_knee, right_ankle),
                (left_elbow, left_shoulder, left_hip),
                (right_elbow, right_shoulder, right_hip)]:
                ang = compute_angle(A, B, C)
                if ang is not None:
                    draw_half_arc_and_label(annotated, A, B, C)
                    record["joint_angles"][f"{int(B[0])}_{int(B[1])}"] = round(ang, 2)

            # --- GRF, Punch etc. (same as before) ---
            def compute_grf(ankle_pt, prev_pt):
                if prev_pt is None: return 0.0, np.array([0, 0])
                vy = (ankle_pt[1] - prev_pt[1])
                Fc = clamp(vy * PIXEL_TO_METER * LEG_MASS * fps, GRF_MIN, GRF_MAX)
                return Fc, np.array([0, int(-Fc * GRF_SCALE_PX)], int)
            Fl, vec_l = compute_grf(left_ankle, prev_ankles["left"])
            Fr, vec_r = compute_grf(right_ankle, prev_ankles["right"])
            prev_ankles["left"], prev_ankles["right"] = left_ankle.copy(), right_ankle.copy()
            draw_arrow_with_label(annotated, left_ankle, vec_l, GRF_COLOR, f"{Fl:.2f}N")
            draw_arrow_with_label(annotated, right_ankle, vec_r, GRF_COLOR, f"{Fr:.2f}N")
            record["forces"]["grf_left"], record["forces"]["grf_right"] = Fl, Fr

            def draw_punch(shoulder, wrist, color=PUNCH_COLOR):
                v = wrist - shoulder
                if np.linalg.norm(v) < 1e-6: return 0.0
                vx = v[0] * PIXEL_TO_METER * fps
                Fp = 5.0 * abs(vx)
                draw_arrow_with_label(annotated, shoulder, (v*0.25).astype(int), color, f"{Fp:.1f}N")
                return Fp
            record["forces"]["punch_left"] = draw_punch(left_shoulder, left_wrist)
            record["forces"]["punch_right"] = draw_punch(right_shoulder, right_wrist)

        # Hand landmarks (extra)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand_landmarks.landmark]
                for (x,y) in pts: cv2.circle(annotated, (x,y), 2, (0,255,255), -1)
                # connect fingers
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(annotated, f"Frame: {frame_idx}", (10,h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1)
        out.write(annotated)
        log_records.append(record)

        cv2.imshow("Annotated Biomech", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
        frame_idx += 1

finally:
    pose.close(); hands.close()
    cap.release(); out.release(); cv2.destroyAllWindows()
    with open(OUTPUT_JSON,"w") as f: json.dump(log_records,f,indent=2)

print("Done. Saved:", OUTPUT_VIDEO, OUTPUT_JSON)

