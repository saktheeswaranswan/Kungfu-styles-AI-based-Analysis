import os
import cv2
import json
import math
import numpy as np
import mediapipe as mp
from collections import OrderedDict

"""
Mediapipe Biomechanics Annotator (Full Rewrite)
------------------------------------------------
Adds:
1) Leg normal vector (yellow) whose magnitude changes with leg angular velocity ("kinetic linking").
2) Reaction force (GRF) arrow from each foot, pointing outward/up from the ankle, with magnitude based on
   estimated vertical acceleration of the ankle (per-foot mass model: ~10 kg each).
3) Joint arcs + variable ellipses at joints (size scales with joint angular speed).
4) Hook punch estimation: estimates force arrow from the wrist based on wrist acceleration * effective hand/forearm mass.
5) Expanded JSON logging of forces, angles, velocities, and accelerations per frame.

Notes:
- Coordinates are in image space (x right, y down). Be mindful of sign conventions.
- GRF model is a simple proxy: F ≈ m_leg * (g - a_y). When ankle accelerates upward (negative a_y), GRF grows.
- Hook punch force: F ≈ m_hand * |a_wrist|, arrow along current wrist velocity.
- Variable ellipses scale with the length of the associated arrow or angular speed.
"""

# =================== CONFIG ===================
VIDEO_IN = "drunkenmonkeystyle.mp4"          # input video file or 0 for webcam
OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/annotated_biomech.mp4"
OUTPUT_JSON = f"{OUTPUT_DIR}/pose_biomech.json"

# Mass model
WEIGHT_KG = 60.0              # whole-body mass (used for legacy displays if needed)
MASS_LEG_KG = 10.0            # per-leg effective mass (~10 kg each) as requested
MASS_HAND_KG = 5.0            # effective mass for hand+forearm for punch model (tunable)
G = 9.81                      # gravity constant (m/s^2)

# Drawing parameters
SCALE_PIXELS_PER_N = 0.030    # global scale for Newton-to-pixels (tune for your video size)
TAIL_LEN_PX = 28
BASE_ARROW_THICKNESS = 3
BASE_TAIL_THICKNESS = 6

# Leg normal (kinetic linking) style
LEG_NORMAL_COLOR = (0, 255, 255)     # yellow
LEG_NORMAL_SCALE = 0.60              # pixels per (deg/s)
LEG_NORMAL_MIN = 15.0                # min length px

# GRF style
GRF_COLOR = (255, 0, 0)              # blue-ish? (BGR) -> here pure red; change if desired
GRF_UP_VECTOR = np.array([0.0, -1.0])

# Hook punch style
PUNCH_COLOR = (0, 0, 255)            # red
PUNCH_TIP = 0.28

# Joint arc + ellipse style
ARC_RADIUS_PX = 45
ARC_THICKNESS = 2
ARC_COLOR = (0, 255, 0)
JOINT_TEXT_COLOR = (255, 200, 0)
ELLIPSE_BASE_A = 30
ELLIPSE_BASE_B = 18
ELLIPSE_MAG_SCALE = 0.10
ELLIPSE_COLOR = (120, 180, 60)

# Skeleton style
LEG_COLOR = (255, 255, 255)
SKELETON_COLOR = (200, 200, 200)

# Misc labels
MAG_TEXT_COLOR = (0, 140, 255)

# EMA smoothing
VEL_EMA_ALPHA = 0.5
ACC_EMA_ALPHA = 0.5
ANGVEL_EMA_ALPHA = 0.5

# ==============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
mp_pose = mp.solutions.pose

# ---------- Utility helpers ----------
def safe_point(landmark, w, h):
    return np.array([float(landmark.x * w), float(landmark.y * h)])

def draw_text_bg(img, text, pos, font_scale=0.5, color=(255,255,255), bg_color=(0,0,0), thickness=1):
    x, y = int(pos[0]), int(pos[1])
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x-3, y-h-3), (x + w + 3, y + 3), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)
    ang = math.degrees(math.acos(cosang))
    return ang

def draw_arrow(img, origin, vec_unit, magnitude_N,
               scale_pixels_per_N=SCALE_PIXELS_PER_N,
               tail_len_px=TAIL_LEN_PX,
               base_arrow_thickness=BASE_ARROW_THICKNESS,
               base_tail_thickness=BASE_TAIL_THICKNESS,
               arrow_color=(0,0,255), tail_color=(0,255,255), tip=0.28, label=None):
    norm = np.linalg.norm(vec_unit)
    if norm < 1e-6:
        return None
    vhat = vec_unit / norm
    length_px = float(magnitude_N) * scale_pixels_per_N
    arrow_th = int(max(1, base_arrow_thickness + length_px / 40))
    tail_th = int(max(1, base_tail_thickness + length_px / 60))
    tail_end = origin + vhat * tail_len_px
    cv2.line(img, tuple(origin.astype(int)), tuple(tail_end.astype(int)), tail_color, tail_th)
    end = origin + vhat * length_px
    cv2.arrowedLine(img, tuple(origin.astype(int)), tuple(end.astype(int)), arrow_color, arrow_th, tipLength=tip)
    if label is not None:
        perp = np.array([-vhat[1], vhat[0]])
        label_pos = end + perp * 10 + vhat * 6
        draw_text_bg(img, label, (int(label_pos[0]), int(label_pos[1])), font_scale=0.55, color=MAG_TEXT_COLOR, bg_color=(20,20,20), thickness=2)
    return {
        "origin": [float(origin[0]), float(origin[1])],
        "end": [float(end[0]), float(end[1])],
        "unit_vec": [float(vhat[0]), float(vhat[1])],
        "length_px": float(length_px),
        "arrow_thickness_px": arrow_th,
        "tail_thickness_px": tail_th
    }

def draw_angle_arc(img, center, leg_unit, down_unit=np.array([0.0, 1.0]), radius_px=ARC_RADIUS_PX, color=ARC_COLOR, thickness=ARC_THICKNESS):
    def norm360(a):
        a = a % 360.0
        return a + 360.0 if a < 0 else a
    ang_leg = norm360(math.degrees(math.atan2(leg_unit[1], leg_unit[0])))
    ang_down = norm360(math.degrees(math.atan2(down_unit[1], down_unit[0])))
    diff = (ang_leg - ang_down + 360.0) % 360.0
    if diff > 180.0:
        start_angle, end_angle = ang_leg, ang_down
    else:
        start_angle, end_angle = ang_down, ang_leg
    center_int = (int(center[0]), int(center[1]))
    cv2.ellipse(img, center_int, (radius_px, radius_px), 0.0, start_angle, end_angle, color, thickness)
    end_rad = math.radians(end_angle)
    arrow_pt = (int(center[0] + radius_px * math.cos(end_rad)), int(center[1] + radius_px * math.sin(end_rad)))
    cv2.circle(img, arrow_pt, 5, color, -1)
    included = abs((ang_leg - ang_down + 360.0) % 360.0)
    if included > 180.0:
        included = 360.0 - included
    return {
        "center":[float(center[0]), float(center[1])],
        "start_angle_deg":float(start_angle),
        "end_angle_deg":float(end_angle),
        "included_angle_deg":float(included),
        "arc_end_point":[float(arrow_pt[0]), float(arrow_pt[1])]
    }

def draw_variable_ellipse(img, center, axis_unit, scale_val, base_a=ELLIPSE_BASE_A, base_b=ELLIPSE_BASE_B, mag_scale=ELLIPSE_MAG_SCALE, color=ELLIPSE_COLOR, thickness=2):
    a = int(round(base_a + scale_val * mag_scale))
    b = int(round(base_b + scale_val * mag_scale * 0.55))
    ang_deg = math.degrees(math.atan2(axis_unit[1], axis_unit[0]))
    center_int = (int(center[0]), int(center[1]))
    cv2.ellipse(img, center_int, (max(1,a), max(1,b)), ang_deg, 0, 360, color, thickness)
    return {
        "center":[float(center[0]), float(center[1])],
        "semi_axis_a_px":float(a),
        "semi_axis_b_px":float(b),
        "rotation_deg":float(ang_deg)
    }

# ---------- Kinematics helpers ----------
class EMA:
    def __init__(self, alpha):
        self.alpha = alpha
        self.val = None
    def update(self, x):
        if self.val is None:
            self.val = x
        else:
            self.val = self.alpha * x + (1 - self.alpha) * self.val
        return self.val

# ---------- Main processing ----------
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {VIDEO_IN}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

pose_connections = [(mp_pose.PoseLandmark(a).name, mp_pose.PoseLandmark(b).name) for a,b in mp_pose.POSE_CONNECTIONS]
pose_data = []
frame_idx = 0

down = np.array([0.0, 1.0])

# Trackers for velocities/accelerations and angles
prev_pts = None
vel_ema = {}
acc_ema = {}
ang_prev = {"LEFT": None, "RIGHT": None}
angvel_ema = {"LEFT": EMA(ANGVEL_EMA_ALPHA), "RIGHT": EMA(ANGVEL_EMA_ALPHA)}

# Define joint triplets (a, b, c) -> angle at b
JOINT_DEFINITIONS = {
    # Left side
    "LEFT_ELBOW": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "LEFT_SHOULDER": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "LEFT_WRIST": ("LEFT_ELBOW", "LEFT_WRIST", "LEFT_INDEX"),
    "LEFT_KNEE": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "LEFT_HIP": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "LEFT_ANKLE": ("LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    # Right side
    "RIGHT_ELBOW": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "RIGHT_SHOULDER": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "RIGHT_WRIST": ("RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_INDEX"),
    "RIGHT_KNEE": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "RIGHT_HIP": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
    "RIGHT_ANKLE": ("RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
}

# Which wrist(s) to use for punch estimation
WRISTS = ["LEFT_WRIST", "RIGHT_WRIST"]

# Helpers for per-landmark EMA containers
def get_ema(container, key, alpha):
    if key not in container:
        container[key] = EMA(alpha)
    return container[key]

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        annotated = frame.copy()

        frame_record = OrderedDict([
            ("frame", int(frame_idx)),
            ("time_s", float(frame_idx / fps)),
            ("keypoints", {}),
            ("edges", pose_connections),
            ("forces", {"leg_normal": {}, "grf": {}, "punch": {}}),
            ("joint_angles", {}),
            ("kinematics", {"velocity": {}, "acceleration": {}, "angular": {}})
        ])

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # collect keypoints
            pts = {}
            for name in mp_pose.PoseLandmark:
                p = lm[name.value]
                pts[name.name] = {
                    "x": float(p.x * width),
                    "y": float(p.y * height),
                    "visibility": float(getattr(p, "visibility", 1.0))
                }
            frame_record["keypoints"] = pts

            # draw skeleton & landmarks
            for a_name, b_name in pose_connections:
                pa, pb = pts[a_name], pts[b_name]
                if pa["visibility"] > 0.1 and pb["visibility"] > 0.1:
                    cv2.line(annotated, (int(pa["x"]), int(pa["y"])), (int(pb["x"]), int(pb["y"])), SKELETON_COLOR, 2)
            for nm, p in pts.items():
                if p["visibility"] > 0.1:
                    cv2.circle(annotated, (int(p["x"]), int(p["y"])), 3, LEG_COLOR, -1)

            # --- Kinematics: velocity & acceleration (EMA smoothed) ---
            dt = 1.0 / float(fps)
            if prev_pts is not None:
                for nm, p in pts.items():
                    if p["visibility"] <= 0.1 or nm not in prev_pts or prev_pts[nm]["visibility"] <= 0.1:
                        continue
                    pos = np.array([p["x"], p["y"]], dtype=np.float32)
                    pos_prev = np.array([prev_pts[nm]["x"], prev_pts[nm]["y"]], dtype=np.float32)
                    v = (pos - pos_prev) / dt
                    v_ema = get_ema(vel_ema, nm, VEL_EMA_ALPHA).update(v)
                    # acceleration (from EMA velocity diffs)
                    a = (v_ema - get_ema(vel_ema, nm, VEL_EMA_ALPHA).val) if False else (v - (pos_prev - np.array([prev_pts[nm]["x"], prev_pts[nm]["y"]]) )/dt)  # not used; proper EMA acc below
                    # compute raw acc from last EMA value if available
                    if nm not in acc_ema:
                        acc_ema[nm] = EMA(ACC_EMA_ALPHA)
                        acc_val = acc_ema[nm].update(np.array([0.0, 0.0], dtype=np.float32))
                    else:
                        # approximate acceleration as dv/dt from raw v (less laggy)
                        v_prev = getattr(get_ema(vel_ema, nm, VEL_EMA_ALPHA), 'prev_v', None)
                        if v_prev is None:
                            dv = np.array([0.0, 0.0], dtype=np.float32)
                        else:
                            dv = (v - v_prev) / dt
                        acc_val = acc_ema[nm].update(dv)
                        get_ema(vel_ema, nm, VEL_EMA_ALPHA).prev_v = v
                    # save
                    frame_record["kinematics"]["velocity"][nm] = [float(v_ema[0]), float(v_ema[1])]
                    frame_record["kinematics"]["acceleration"][nm] = [float(acc_val[0]), float(acc_val[1])]

            # --- Leg normal (yellow) + angle arc, per leg ---
            for side in ["LEFT", "RIGHT"]:
                hip_key = f"{side}_HIP"
                ankle_key = f"{side}_ANKLE"
                if hip_key not in pts or ankle_key not in pts:
                    continue
                if pts[hip_key]["visibility"] <= 0.1 or pts[ankle_key]["visibility"] <= 0.1:
                    continue

                hip = np.array([pts[hip_key]["x"], pts[hip_key]["y"]], dtype=np.float32)
                ankle = np.array([pts[ankle_key]["x"], pts[ankle_key]["y"]], dtype=np.float32)
                leg_vec = ankle - hip
                if np.linalg.norm(leg_vec) < 1e-6:
                    continue
                leg_unit = leg_vec / np.linalg.norm(leg_vec)

                # Angle from vertical for reference
                cos_theta = np.dot(leg_unit, down) / (np.linalg.norm(down) * np.linalg.norm(leg_unit))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta_rad = math.acos(cos_theta)
                theta_deg = math.degrees(theta_rad)

                # Angular velocity of leg (change of angle wrt previous frame)
                ang_now = theta_deg
                if ang_prev[side] is None:
                    ang_prev[side] = ang_now
                ang_vel = (ang_now - ang_prev[side]) / max(dt, 1e-6)  # deg/s
                ang_prev[side] = ang_now
                ang_vel_s = angvel_ema[side].update(ang_vel)

                # Leg normal vector (perpendicular to leg axis). Flip to point roughly upward/outward.
                perp = np.array([-leg_unit[1], leg_unit[0]], dtype=np.float32)
                if np.dot(perp, np.array([0.0, -1.0])) < 0:  # try to point somewhat upward
                    perp = -perp

                # Length scales with |angular velocity|
                leg_normal_len = LEG_NORMAL_MIN + LEG_NORMAL_SCALE * abs(ang_vel_s)
                leg_normal_end = ankle + perp * leg_normal_len
                cv2.arrowedLine(annotated, tuple(ankle.astype(int)), tuple(leg_normal_end.astype(int)), LEG_NORMAL_COLOR, 2, tipLength=0.25)

                # Draw arc to vertical + label
                arc_info = draw_angle_arc(annotated, ankle, leg_unit, down_unit=down, radius_px=ARC_RADIUS_PX, color=ARC_COLOR, thickness=ARC_THICKNESS)
                draw_text_bg(annotated, f"{theta_deg:.1f}°", (int(ankle[0]+8), int(ankle[1] - ARC_RADIUS_PX - 8)), font_scale=0.6, color=ARC_COLOR, bg_color=(10,10,10), thickness=2)

                # Variable ellipse around ankle scaled by leg angular speed
                ell = draw_variable_ellipse(annotated, ankle, leg_unit, scale_val=abs(ang_vel_s))

                # Save
                frame_record["forces"]["leg_normal"][side.lower()] = {
                    "ankle": [float(ankle[0]), float(ankle[1])],
                    "hip": [float(hip[0]), float(hip[1])],
                    "angle_from_vertical_deg": float(theta_deg),
                    "angular_velocity_deg_s": float(ang_vel_s),
                    "normal_vector_px": [float(perp[0] * leg_normal_len), float(perp[1] * leg_normal_len)],
                    "arc_info": arc_info,
                    "ellipse_info": ell,
                }
                frame_record["kinematics"]["angular"][f"{side}_leg_angle_deg"] = float(theta_deg)
                frame_record["kinematics"]["angular"][f"{side}_leg_ang_vel_deg_s"] = float(ang_vel_s)

            # --- Ground Reaction Force (GRF) from ankle vertical acceleration ---
            for side in ["LEFT", "RIGHT"]:
                ankle_key = f"{side}_ANKLE"
                if ankle_key not in pts or pts[ankle_key]["visibility"] <= 0.1:
                    continue
                if ankle_key not in acc_ema or ankle_key not in vel_ema:
                    continue  # wait until we have kinematics

                ankle = np.array([pts[ankle_key]["x"], pts[ankle_key]["y"]], dtype=np.float32)
                # acceleration y-component (pixels/s^2). We'll use sign logic: y down -> upward acceleration is negative.
                ay = acc_ema[ankle_key].val[1] if acc_ema[ankle_key].val is not None else 0.0

                # Scale pixels to meters? Unknown without calibration. We'll treat as relative magnitude.
                # We still compute a pseudo-GRF in Newtons using MASS_LEG_KG * (g - a_y_scaled).
                # Without calibration, set a_y_scaled = ay * K where K is tiny so GRF doesn't explode.
                PIXEL_TO_METER = 1/600.0  # crude guess: 600 px ~ 1 m (adjust as needed)
                a_y_scaled = ay * PIXEL_TO_METER
                grf_N = MASS_LEG_KG * max(0.0, (G - a_y_scaled))

                info = draw_arrow(
                    annotated,
                    origin=ankle,
                    vec_unit=GRF_UP_VECTOR,
                    magnitude_N=grf_N,
                    arrow_color=GRF_COLOR,
                    tail_color=(255,255,255),
                    tip=0.28,
                    label=f"GRF {grf_N:.0f} N"
                )

                frame_record["forces"]["grf"][side.lower()] = {
                    "ankle": [float(ankle[0]), float(ankle[1])],
                    "ay_pixels_s2": float(ay),
                    "ay_m_s2_est": float(a_y_scaled),
                    "mass_leg_kg": float(MASS_LEG_KG),
                    "grf_N": float(grf_N),
                    "arrow": info
                }

            # --- Hook punch force (per wrist) ---
            for wrist_key in WRISTS:
                if wrist_key not in pts or pts[wrist_key]["visibility"] <= 0.1:
                    continue
                if wrist_key not in vel_ema or wrist_key not in acc_ema:
                    continue
                wrist = np.array([pts[wrist_key]["x"], pts[wrist_key]["y"]], dtype=np.float32)
                v = vel_ema[wrist_key].val if vel_ema[wrist_key].val is not None else np.array([0.0, 0.0], dtype=np.float32)
                a = acc_ema[wrist_key].val if acc_ema[wrist_key].val is not None else np.array([0.0, 0.0], dtype=np.float32)

                vnorm = np.linalg.norm(v)
                anorm = np.linalg.norm(a)
                if vnorm < 1e-6:
                    continue
                vhat = v / max(vnorm, 1e-6)

                # Convert pixel/s^2 to m/s^2 using the same crude scale
                a_scaled = a * (1/600.0)
                a_scaled_norm = float(np.linalg.norm(a_scaled))

                punch_F = MASS_HAND_KG * a_scaled_norm  # N
                info = draw_arrow(
                    annotated,
                    origin=wrist,
                    vec_unit=vhat,
                    magnitude_N=punch_F,
                    arrow_color=PUNCH_COLOR,
                    tail_color=(255,255,255),
                    tip=PUNCH_TIP,
                    label=f"Punch {punch_F:.0f} N"
                )

                # Draw local ellipse oriented along velocity, scaled by |v|
                vel_scale = float(vnorm * 0.02)  # px scaling for ellipse variety
                ell = draw_variable_ellipse(annotated, wrist, vhat, scale_val=vel_scale)

                frame_record["forces"]["punch"][wrist_key.lower()] = {
                    "wrist": [float(wrist[0]), float(wrist[1])],
                    "vel_px_s": [float(v[0]), float(v[1])],
                    "acc_px_s2": [float(a[0]), float(a[1])],
                    "acc_m_s2_est": [float(a_scaled[0]), float(a_scaled[1])],
                    "mass_hand_kg": float(MASS_HAND_KG),
                    "force_N": float(punch_F),
                    "arrow": info,
                    "ellipse_info": ell
                }

            # --- Joint angles + small arcs (also variable ellipses scaled with ang speed) ---
            for joint_name, (a_name, b_name, c_name) in JOINT_DEFINITIONS.items():
                pa = pts.get(a_name)
                pb = pts.get(b_name)
                pc = pts.get(c_name)
                if pa is None or pb is None or pc is None:
                    continue
                if pa["visibility"] < 0.1 or pb["visibility"] < 0.1 or pc["visibility"] < 0.1:
                    continue

                A = np.array([pa["x"], pa["y"]])
                B = np.array([pb["x"], pb["y"]])
                C = np.array([pc["x"], pc["y"]])

                angle = compute_angle(A, B, C)
                if angle is None:
                    continue

                # draw angle text near joint B
                text_pos = (int(B[0] + 6), int(B[1] - 6))
                draw_text_bg(annotated, f"{angle:.1f}°", text_pos, font_scale=0.5, color=JOINT_TEXT_COLOR, bg_color=(10,10,10), thickness=1)

                # small arc representing angle between BA and BC
                v1 = A - B
                v2 = C - B
                a1 = math.degrees(math.atan2(v1[1], v1[0]))
                a2 = math.degrees(math.atan2(v2[1], v2[0]))
                arc_r = 20
                start_ang = a1 % 360
                end_ang = a2 % 360
                diff = (end_ang - start_ang + 360) % 360
                if diff > 180:
                    start_angle_for_cv = end_ang
                    end_angle_for_cv = start_ang
                else:
                    start_angle_for_cv = start_ang
                    end_angle_for_cv = end_ang
                try:
                    cv2.ellipse(annotated, (int(B[0]), int(B[1])), (arc_r, arc_r), 0.0, start_angle_for_cv, end_angle_for_cv, JOINT_TEXT_COLOR, 1)
                except Exception:
                    pass

                # Variable ellipse at joint scaled by approximate angular speed (via change in angle over dt if we can cache)
                key = f"{joint_name}_prev_angle"
                if key not in ang_prev:
                    ang_prev[key] = angle
                    ang_rate = 0.0
                else:
                    ang_rate = (angle - ang_prev[key]) / max(dt, 1e-6)  # deg/s
                    ang_prev[key] = angle
                # ellipse oriented along bisector between v1 and v2
                v1u = v1 / (np.linalg.norm(v1) + 1e-6)
                v2u = v2 / (np.linalg.norm(v2) + 1e-6)
                axis = v1u + v2u
                if np.linalg.norm(axis) < 1e-6:
                    axis = v1u  # fallback
                axis = axis / (np.linalg.norm(axis) + 1e-6)
                ell = draw_variable_ellipse(annotated, B, axis, scale_val=abs(ang_rate))

                frame_record["joint_angles"][joint_name] = float(round(angle, 3))

        else:
            cv2.putText(annotated, "No pose detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # frame counter
        draw_text_bg(annotated, f"Frame: {frame_idx}", (12, height - 10), font_scale=0.6, color=(230,230,230), bg_color=(10,10,10), thickness=1)

        out.write(annotated)
        pose_data.append(frame_record)
        prev_pts = frame_record["keypoints"] if frame_record["keypoints"] else prev_pts
        frame_idx += 1

cap.release()
out.release()

# save JSON with all info
with open(OUTPUT_JSON, "w") as f:
    json.dump(pose_data, f, indent=2)

print("✅ Done.")
print(f"Annotated video: {OUTPUT_VIDEO}")
print(f"Pose + biomechanics JSON: {OUTPUT_JSON}")

