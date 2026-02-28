import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import time
import math
import platform
from dataclasses import dataclass
# ===== Hand Lock System =====
_last_wrist_lock = None   # (x,y) gespeicherte Handposition
LOCK_MAX_DIST = 220       # Pixel — wie weit eine neue Hand entfernt sein darf
LOCK_TIMEOUT = 1.2        # Sekunden ohne Sicht -> lock wird gelöscht
_last_seen_time = 0
from typing import Dict, Optional, List

import cv2
import mediapipe as mp

# Classic API (bei korrekter mediapipe Installation vorhanden)
mp_solutions = mp.solutions

# UART optional (nur benutzen wenn uart_enabled=True)
try:
    from uart_mod import uart
except Exception:
    uart = None


# ============================================================
# CONFIG
# ============================================================
# Du willst NUR die rechte Hand (MediaPipe Label "Right") tracken.
# Daher: KEIN Swap beim Flip.
SWAP_HANDEDNESS_WHEN_FLIP = False


# ============================================================
# Dataclasses
# ============================================================
@dataclass
class FingerAngles:
    mcp: float
    pip: float
    dip: float
    curl: float


@dataclass
class HandKinematics:
    yaw: float
    pitch: float
    roll: float

    wrist_px: int
    wrist_py: int

    # Index PIP (2. Gelenk) als Pixel
    index_pip_px: int
    index_pip_py: int

    fingers: Dict[str, FingerAngles]


# ============================================================
# Math helpers
# ============================================================
def _vec(a, b):
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])

def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

def _cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )

def _norm(u):
    return math.sqrt(_dot(u, u)) + 1e-9

def _angle_deg(u, v):
    nu = _norm(u)
    nv = _norm(v)
    cosang = _dot(u, v) / (nu * nv)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

def _safe_int(x, lo, hi):
    return int(max(lo, min(hi, x)))


# ============================================================
# Core extraction
# ============================================================
def extract_kinematics_for_right_hand(results, w: int, h: int, flip: bool = True) -> Optional[HandKinematics]:
    global _last_wrist_lock, _last_seen_time

    if results.multi_hand_landmarks is None or results.multi_handedness is None:
        # timeout reset
        if time.time() - _last_seen_time > LOCK_TIMEOUT:
            _last_wrist_lock = None
        return None

    # ------------------------------------------------------------
    # FIX: Fokus NUR auf rechte Hand
    # -> desired_label bleibt immer "Right"
    # -> kein Swap auf "Left" beim flip
    # ------------------------------------------------------------
    desired_label = "Right"

    candidates = []

    # ---- sammle nur rechte Hände
    for hand_lms, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handed.classification[0].label
        if label != desired_label:
            continue

        wrist = hand_lms.landmark[0]
        px = int(wrist.x * w)
        py = int(wrist.y * h)
        candidates.append((hand_lms, px, py))

    if len(candidates) == 0:
        if time.time() - _last_seen_time > LOCK_TIMEOUT:
            _last_wrist_lock = None
        return None

    # ---- Wenn noch kein Lock: nimm die erste
    if _last_wrist_lock is None:
        best = candidates[0]
    else:
        # nimm die Hand die am nächsten zur letzten Position ist
        best = None
        best_dist = 999999

        for c in candidates:
            _, px, py = c
            dx = px - _last_wrist_lock[0]
            dy = py - _last_wrist_lock[1]
            dist = (dx*dx + dy*dy)

            if dist < best_dist:
                best_dist = dist
                best = c

        # Wenn plötzlich eine GANZ andere Hand erscheint -> ignorieren
        if best_dist > LOCK_MAX_DIST * LOCK_MAX_DIST:
            return None

    hand_lms, wrist_px, wrist_py = best

    _last_wrist_lock = (wrist_px, wrist_py)
    _last_seen_time = time.time()

    # ----- ab hier original code
    lm = []
    for i in range(21):
        x = hand_lms.landmark[i].x
        y = hand_lms.landmark[i].y
        z = hand_lms.landmark[i].z
        lm.append((x, y, z))

    index_pip = lm[6]
    index_pip_px = int(index_pip[0] * w)
    index_pip_py = int(index_pip[1] * h)

    v1 = _vec(lm[0], lm[5])
    v2 = _vec(lm[0], lm[17])
    nx, ny, nz = _cross(v1, v2)
    yaw = math.degrees(math.atan2(nx, nz + 1e-9))
    pitch = math.degrees(math.atan2(ny, nz + 1e-9))

    i5 = (lm[5][0] * w, lm[5][1] * h)
    p17 = (lm[17][0] * w, lm[17][1] * h)
    roll = math.degrees(math.atan2((p17[1] - i5[1]), (p17[0] - i5[0] + 1e-9)))

    finger_map = {
        "thumb": (1, 2, 3, 4),
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }

    fingers: Dict[str, FingerAngles] = {}
    for name, (mcp_i, pip_i, dip_i, tip_i) in finger_map.items():
        wrist_v = _vec(lm[0], lm[mcp_i])
        mcp_v1 = _vec(lm[mcp_i], lm[pip_i])
        pip_v1 = _vec(lm[pip_i], lm[dip_i])
        dip_v1 = _vec(lm[dip_i], lm[tip_i])

        mcp_ang = _angle_deg(tuple(-x for x in wrist_v), mcp_v1)
        pip_ang = _angle_deg(tuple(-x for x in mcp_v1), pip_v1)
        dip_ang = _angle_deg(tuple(-x for x in pip_v1), dip_v1)

        curl = (pip_ang + dip_ang) / 2.0
        fingers[name] = FingerAngles(mcp=mcp_ang, pip=pip_ang, dip=dip_ang, curl=curl)

    return HandKinematics(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        wrist_px=wrist_px,
        wrist_py=wrist_py,
        index_pip_px=index_pip_px,
        index_pip_py=index_pip_py,
        fingers=fingers,
    )

def draw_overlay(frame, kin: HandKinematics):
    cv2.putText(frame, f"yaw={kin.yaw:+.1f} pitch={kin.pitch:+.1f} roll={kin.roll:+.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.circle(frame, (kin.index_pip_px, kin.index_pip_py), 6, (0, 255, 255), -1)
    cv2.putText(frame, "index_pip", (kin.index_pip_px + 8, kin.index_pip_py - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# ============================================================
# Camera utilities
# ============================================================
def _try_open(index: int, backend: Optional[int] = None):
    cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

def list_available_cameras(max_index: int = 6) -> List[int]:
    sysn = platform.system().lower()
    backends = [None]
    if "windows" in sysn:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    elif "linux" in sysn:
        backends = [cv2.CAP_V4L2, None]

    available = []
    for i in range(max_index):
        opened = False
        for be in backends:
            cap = _try_open(i, be)
            if cap is not None:
                opened = True
                cap.release()
                break
        if opened:
            available.append(i)
    return available

def open_best_camera(preferred_index: int, max_index: int = 6):
    sysn = platform.system().lower()
    backends = [None]
    if "windows" in sysn:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    elif "linux" in sysn:
        backends = [cv2.CAP_V4L2, None]

    for be in backends:
        cap = _try_open(preferred_index, be)
        if cap is not None:
            return cap, preferred_index, be

    for i in range(max_index):
        for be in backends:
            cap = _try_open(i, be)
            if cap is not None:
                return cap, i, be

    raise RuntimeError("Keine Kamera gefunden. Prüfe Rechte / ob Kamera belegt ist / max_index erhöhen.")


# ============================================================
# Tracker
# ============================================================
class RightHandAngleTracker:
    def __init__(
        self,
        camera_index: int = 0,
        flip: bool = True,
        max_camera_index_to_scan: int = 6,
        model_complexity: int = 1,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        target_w: int = 1280,
        target_h: int = 720,
        uart_enabled: bool = False,
        uart_period: float = 0.08,
    ):
        self.flip = flip
        self.target_w = target_w
        self.target_h = target_h

        self.cap, chosen_idx, backend = open_best_camera(camera_index, max_camera_index_to_scan)
        print(f"[Camera] chosen_index={chosen_idx} backend={backend}")

        # try to request resolution (may be ignored by some cams)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.hands = mp_solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=model_complexity,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # UART
        self.uart_enabled = uart_enabled and (uart is not None)
        self.uart_period = uart_period
        self._last_uart_send = 0.0
        self._uart = uart() if self.uart_enabled else None

        self._last_kin: Optional[HandKinematics] = None
        self._last_frame = None

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass

    def update(self, draw: bool = False):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._last_kin = None
            self._last_frame = None
            return None

        if self.flip:
            frame = cv2.flip(frame, 1)

        # Force consistent size (cams may ignore CAP_PROP settings)
        if frame.shape[1] != self.target_w or frame.shape[0] != self.target_h:
            frame = cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        kin = extract_kinematics_for_right_hand(results, w, h, flip=self.flip)
        self._last_kin = kin
        self._last_frame = frame

        # UART output only when enabled (HandOnly)
        if self.uart_enabled and kin is not None:
            now = time.time()
            if (now - self._last_uart_send) > self.uart_period:
                def _clamp(v, lo=0.0, hi=180.0):
                    return int(max(lo, min(hi, v)))

                self._uart.p0 = _clamp(kin.fingers["thumb"].curl)
                self._uart.p1 = _clamp(kin.fingers["index"].curl)
                self._uart.p2 = _clamp(kin.fingers["middle"].curl)
                self._uart.p3 = _clamp(kin.fingers["ring"].curl)
                self._uart.p4 = _clamp(kin.fingers["pinky"].curl)

                try:
                    self._uart.send()
                except Exception as e:
                    print(f"[UART] ERROR in send(): {e} -> UART wird deaktiviert")
                    self.uart_enabled = False
                    self._uart = None
                self._last_uart_send = now

        if draw and kin is not None:
            draw_overlay(frame, kin)

        return frame

    def has_hand(self) -> bool:
        return self._last_kin is not None

    def get_last_kin(self) -> Optional[HandKinematics]:
        return self._last_kin
