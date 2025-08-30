import cv2
import mediapipe as mp
import numpy as np
import json
import math
import time
from collections import OrderedDict

# ---------- CONFIG ----------
VIDEO_IN = "drunkenmonkeystyle.mp4"   # or 0 for webcam
OUTPUT_VIDEO = "output/annotated_output.mp4"
OUTPUT_JSON = "output/motion_log.json"
FOURCC = "mp4v"
PIXEL_TO_METER = 0.002   # crude scale if you want to compute forces
LEG_MASS = 10.0

# visual params
LEG_NORMAL_COLOR = (0, 255, 255)   # yellow
GRF_COLOR = (255, 0, 0)            # blue
PUNCH_COLOR = (0, 0, 255)          # red
ARC_COLOR = (0, 255, 0)            # green for arcs
ANGLE_TEXT_COLOR = (255, 200, 0)   # yellow-ish
LANDMARK_COLOR = (255, 255, 255)   # white
EDGE_COLOR = (200, 200, 200)       # grey edges for skeleton
FPS_FALLBACK = 25

# GRF variation clamp
GRF_MIN = -0.6
GRF_MAX = 0.6
GRF_SCALE_PX = 50

# --------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# helpers
def clamp(x, a, b):
    return max(a, min(b, x))

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def draw_half_arc_and_label(img, A, B, C, color=ARC_COLOR, radius_px=30):
    v1 = A - B
    v2 = C - B
    if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
        return
    ang1 = math.degrees(math.atan2(v1[1], v1[0])) % 360
    ang2 = math.degrees(math.atan2(v2[1], v2[0])) % 360
    diff = (ang2 - ang1 + 360) % 360
    if diff > 180:
        start_ang, end_ang = ang2, ang1
    else:
        start_ang, end_ang = ang1, ang2
    center = (int(B[0]), int(B[1]))
    try:
        cv2.ellipse(img, center, (radius_px, radius_px), 0.0, start_ang, end_ang, color, 2)
    except Exception:
        try:
            cv2.ellipse(img, center, (radius_px, radius_px), 0.0, min(start_ang, end_ang), max(start_ang, end_ang), color, 2)
        except Exception:
            pass
    included = abs((math.degrees(math.atan2(v2[1], v2[0])) - math.degrees(math.atan2(v1[1], v1[0])) + 360) % 360)
    if included > 180:
        included = 360 - included
    label_pos = (center[0] + 8, center[1] - radius_px - 6)
    cv2.putText(img, f"{included:.0f}°", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANGLE_TEXT_COLOR, 1, cv2.LINE_AA)

def draw_arrow_with_label(img, origin, vec_px, color=(255,0,0), label=None, thickness=2):
    origin_pt = (int(origin[0]), int(origin[1]))
    end_pt = (int(origin[0] + vec_px[0]), int(origin[1] + vec_px[1]))
    cv2.arrowedLine(img, origin_pt, end_pt, color, thickness, tipLength=0.25)
    if label is not None:
        perp = np.array([-vec_px[1], vec_px[0]])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.array([0, -1])
        perp = perp / (np.linalg.norm(perp) + 1e-6)
        label_pos = (int(end_pt[0] + perp[0]*8), int(end_pt[1] + perp[1]*8))
        cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANGLE_TEXT_COLOR, 1, cv2.LINE_AA)

# Open video
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise SystemExit(f"Cannot open input {VIDEO_IN}")

fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*FOURCC)
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

frame_idx = 0
prev_ankles = {"left": None, "right": None}
log_records = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        annotated = frame.copy()

        record = OrderedDict([
            ("frame", int(frame_idx)),
            ("time_s", float(frame_idx / fps)),
            ("keypoints", {}),
            ("forces", {}),
            ("joint_angles", {})
        ])

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def L(i):
                return np.array([float(lm[i].x * w), float(lm[i].y * h)])

            # main landmarks
            left_shoulder = L(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            right_shoulder = L(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            left_elbow = L(mp_pose.PoseLandmark.LEFT_ELBOW.value)
            right_elbow = L(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            left_wrist = L(mp_pose.PoseLandmark.LEFT_WRIST.value)
            right_wrist = L(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            left_hip = L(mp_pose.PoseLandmark.LEFT_HIP.value)
            right_hip = L(mp_pose.PoseLandmark.RIGHT_HIP.value)
            left_knee = L(mp_pose.PoseLandmark.LEFT_KNEE.value)
            right_knee = L(mp_pose.PoseLandmark.RIGHT_KNEE.value)
            left_ankle = L(mp_pose.PoseLandmark.LEFT_ANKLE.value)
            right_ankle = L(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

            # draw landmarks
            for p in (left_shoulder, right_shoulder, left_elbow, right_elbow,
                      left_wrist, right_wrist, left_hip, right_hip,
                      left_knee, right_knee, left_ankle, right_ankle):
                cv2.circle(annotated, (int(p[0]), int(p[1])), 3, LANDMARK_COLOR, -1)

            # --- NEW: connect edges (skeleton) ---
            edges = [
                (left_shoulder, right_shoulder),
                (left_shoulder, left_elbow), (left_elbow, left_wrist),
                (right_shoulder, right_elbow), (right_elbow, right_wrist),
                (left_shoulder, left_hip), (right_shoulder, right_hip),
                (left_hip, right_hip),
                (left_hip, left_knee), (left_knee, left_ankle),
                (right_hip, right_knee), (right_knee, right_ankle)
            ]
            for a, b in edges:
                cv2.line(annotated, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), EDGE_COLOR, 2)

            # --- Joint half-arcs ---
            joints_to_draw = [
                (left_shoulder, left_elbow, left_wrist),
                (right_shoulder, right_elbow, right_wrist),
                (left_hip, left_knee, left_ankle),
                (right_hip, right_knee, right_ankle),
                (left_elbow, left_shoulder, left_hip),
                (right_elbow, right_shoulder, right_hip),
            ]
            for A, B, C in joints_to_draw:
                ang = compute_angle(A, B, C)
                if ang is not None:
                    draw_half_arc_and_label(annotated, A, B, C, color=ARC_COLOR, radius_px=30)
                    record["joint_angles"][f"{int(B[0])}_{int(B[1])}"] = float(round(ang, 2))

            # --- Leg normals ---
            def draw_leg_normal(hip, knee, ankle):
                leg_vec = ankle - hip
                if np.linalg.norm(leg_vec) < 1e-6:
                    return
                perp = np.array([-leg_vec[1], leg_vec[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-6)
                p1 = knee + perp * 40
                draw_arrow_with_label(annotated, knee, p1 - knee, color=LEG_NORMAL_COLOR)
            draw_leg_normal(left_hip, left_knee, left_ankle)
            draw_leg_normal(right_hip, right_knee, right_ankle)

            # --- GRF ---
            def compute_grf(ankle_pt, prev_pt):
                if prev_pt is None:
                    return 0.0, np.array([0, 0])
                vy = (ankle_pt[1] - prev_pt[1])
                F_est = vy * PIXEL_TO_METER * LEG_MASS * fps
                Fc = clamp(F_est, GRF_MIN, GRF_MAX)
                arrow_px = int(-Fc * GRF_SCALE_PX)
                return float(Fc), np.array([0, arrow_px], dtype=int)

            Fl, vec_l = compute_grf(left_ankle, prev_ankles["left"])
            Fr, vec_r = compute_grf(right_ankle, prev_ankles["right"])
            prev_ankles["left"], prev_ankles["right"] = left_ankle.copy(), right_ankle.copy()
            draw_arrow_with_label(annotated, left_ankle, vec_l, color=(255,0,0), label=f"{Fl:.2f}N")
            draw_arrow_with_label(annotated, right_ankle, vec_r, color=(255,0,0), label=f"{Fr:.2f}N")
            record["forces"]["grf_left"], record["forces"]["grf_right"] = Fl, Fr

            # --- Punch ---
            def draw_punch(shoulder_pt, wrist_pt, color=PUNCH_COLOR):
                v = wrist_pt - shoulder_pt
                if np.linalg.norm(v) < 1e-6:
                    return 0.0
                vx = v[0] * PIXEL_TO_METER * fps
                Fp = 5.0 * abs(vx)
                arrow_px = (v * 0.25).astype(int)
                draw_arrow_with_label(annotated, shoulder_pt, arrow_px, color=color, label=f"{Fp:.1f}N")
                return float(Fp)
            Fp_l = draw_punch(left_shoulder, left_wrist, color=PUNCH_COLOR)
            Fp_r = draw_punch(right_shoulder, right_wrist, color=PUNCH_COLOR)
            record["forces"]["punch_left"], record["forces"]["punch_right"] = Fp_l, Fp_r

            record["keypoints"] = {
                "left_shoulder": [float(left_shoulder[0]), float(left_shoulder[1])],
                "right_shoulder": [float(right_shoulder[0]), float(right_shoulder[1])],
                "left_hip": [float(left_hip[0]), float(left_hip[1])],
                "right_hip": [float(right_hip[0]), float(right_hip[1])],
            }

        else:
            cv2.putText(annotated, "No pose detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.putText(annotated, f"Frame: {frame_idx}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1)
        out.write(annotated)
        log_records.append(record)

        cv2.imshow("Annotated Biomech", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

finally:
    pose.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    with open(OUTPUT_JSON, "w") as f:
        json.dump(log_records, f, indent=2)

print("Done. Saved:", OUTPUT_VIDEO, OUTPUT_JSON)

