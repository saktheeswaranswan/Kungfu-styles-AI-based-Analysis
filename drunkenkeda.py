import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

PIXEL_TO_METER = 0.002
LEG_MASS = 2

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture('drunkenmonkeystyle.mp4')

prev_landmarks = None
prev_velocities = None
prev_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w, _ = image.shape
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        joints = {
            'left_ankle': lm[mp_pose.PoseLandmark.LEFT_ANKLE],
            'right_ankle': lm[mp_pose.PoseLandmark.RIGHT_ANKLE],
            'left_knee': lm[mp_pose.PoseLandmark.LEFT_KNEE],
            'right_knee': lm[mp_pose.PoseLandmark.RIGHT_KNEE],
            'left_hip': lm[mp_pose.PoseLandmark.LEFT_HIP],
            'right_hip': lm[mp_pose.PoseLandmark.RIGHT_HIP],
            'left_wrist': lm[mp_pose.PoseLandmark.LEFT_WRIST],
            'right_wrist': lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        }

        joints_px = {k: (int(v.x * w), int(v.y * h)) for k, v in joints.items()}

        if prev_landmarks is not None:
            dt = 1/30.0
            velocities = {}
            accelerations = {}
            for k in joints:
                vx = (joints[k].x - prev_landmarks[k].x) * w / dt
                vy = (joints[k].y - prev_landmarks[k].y) * h / dt
                velocities[k] = np.array([vx, vy])

                if prev_velocities is not None:
                    ax = (vx - prev_velocities[k][0]) / dt
                    ay = (vy - prev_velocities[k][1]) / dt
                    accelerations[k] = np.array([ax, ay])
                else:
                    accelerations[k] = np.array([0, 0])

            # === GRF (strictly vertical arrows) ===
            for side in ['left', 'right']:
                ankle = joints_px[f'{side}_ankle']
                acc_y = -accelerations[f'{side}_ankle'][1] * PIXEL_TO_METER
                F = LEG_MASS * acc_y
                arrow_len = int(F * 50)
                cv2.arrowedLine(image, ankle, (ankle[0], ankle[1] - arrow_len), (255, 0, 0), 3)
                cv2.putText(image, f"GRF {side}: {F:.1f}N", (ankle[0]+10, ankle[1]-arrow_len),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # === Leg normal vectors ===
            for side in ['left', 'right']:
                hip = np.array(joints_px[f'{side}_hip'])
                ankle = np.array(joints_px[f'{side}_ankle'])
                vec = ankle - hip
                normal = np.array([-vec[1], vec[0]])
                length = np.linalg.norm(velocities[f'{side}_hip']) * 0.05
                if np.linalg.norm(normal) > 0:
                    normal = normal / np.linalg.norm(normal)
                p1 = tuple(hip.astype(int))
                p2 = tuple((hip + normal * length).astype(int))
                cv2.arrowedLine(image, p1, p2, (0, 255, 255), 2)

            # === Hook punch force ===
            for side in ['left', 'right']:
                wrist = joints_px[f'{side}_wrist']
                acc = accelerations[f'{side}_wrist'] * PIXEL_TO_METER
                punch_force = 5.0 * np.linalg.norm(acc)
                end_pt = (int(wrist[0] + acc[0]*0.1), int(wrist[1] + acc[1]*0.1))
                cv2.arrowedLine(image, wrist, end_pt, (0, 0, 255), 2)
                cv2.putText(image, f"Punch {side}: {punch_force:.1f}N", (wrist[0]+10, wrist[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            # === Half-arc joint visualization ===
            def draw_half_arc(p1, joint, p2, color=(0,255,0)):
                a = np.array(p1) - np.array(joint)
                b = np.array(p2) - np.array(joint)
                ang1 = math.degrees(math.atan2(a[1], a[0]))
                ang2 = math.degrees(math.atan2(b[1], b[0]))
                angle = (ang2 - ang1) % 360
                center = tuple(map(int, joint))
                cv2.ellipse(image, center, (30,30), 0, ang1, ang2, color, 2)
                cv2.putText(image, f"{int(angle)}°", (center[0]+5, center[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            draw_half_arc(joints_px['left_hip'], joints_px['left_knee'], joints_px['left_ankle'])
            draw_half_arc(joints_px['right_hip'], joints_px['right_knee'], joints_px['right_ankle'])
            draw_half_arc(joints_px['left_shoulder'] if 'left_shoulder' in joints_px else (0,0),
                          joints_px['left_elbow'] if 'left_elbow' in joints_px else (0,0),
                          joints_px['left_wrist'])

            prev_velocities = velocities
        prev_landmarks = joints

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Biomech Annotator', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()

