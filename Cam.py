import os
import time
import glob
import platform
import threading
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

# ============================================================
# UART (GENAU WIE IN Han1dOnly.py)
# ============================================================
try:
    import uart_mod
    UART_AVAILABLE = True
except Exception as e:
    print("\n[WARN] uart_mod nicht verfügbar -> Script läuft ohne UART senden.")
    print("Fehler:", repr(e))
    UART_AVAILABLE = False


# =========================
# SETTINGS
# =========================
GLASS_NAME = "glass"
NEG_NAME = "not_glass"
ARM_NAME = "arm"

GLASS_DATASET_DIR = f"dataset_{GLASS_NAME}"
GLASS_TEMPLATES_DIR = f"templates_{GLASS_NAME}"

NEG_DATASET_DIR = f"dataset_{NEG_NAME}"
NEG_TEMPLATES_DIR = f"templates_{NEG_NAME}"

ARM_DATASET_DIR = f"dataset_{ARM_NAME}"
ARM_TEMPLATES_DIR = f"templates_{ARM_NAME}"

WINDOW_NAME = "CAM (FULL FRAME) | green=glass, blue=arm | ROI=open-latch | overlap=close-latch"

PREFERRED_CAMERA_INDEX = 0
MAX_CAMERA_INDEX_TO_SCAN = 10
TARGET_W = 1280
TARGET_H = 720
FLIP = False

# Speed settings
FRAME_DOWNSCALE = 0.35
SCALES = [1.0]
USE_EDGES = False

# >>> WICHTIG: GANZES BILD ANALYSIEREN
SEARCH_ROI_X_START_RATIO = 0.0  # 0.0 => full frame

# Detection frequency
DETECT_HZ = 15

# Thresholds
GLASS_THRESHOLD = 0.7
GLASS_SCORE_MARGIN = 0.08

ARM_THRESHOLD = 0.7
ARM_SCORE_MARGIN = 0.08

TOPK_POS = 8
MIN_CANDIDATE_DIST = 40

MAX_FILES_PER_BANK = 10000
MAX_VARIANTS_PER_BANK = 20000

# -------------------------
# Hand behavior
# -------------------------
CLOSE_DELAY_S = 1.0

# ROI: wenn Arm-Mittelpunkt drin -> "greifen" (du wolltest tauschen)
ROI_REL_X = 0.05
ROI_REL_Y = 0.20
ROI_REL_W = 0.35
ROI_REL_H = 0.60

# Intersection rule
REQUIRE_MIN_INTERSECTION_AREA = 1

# ============================================================
# SERVO: 0 = geschlossen, 180 = offen
# ============================================================
SERVO_MIN = 0
SERVO_MAX = 180

# Wenn Servo mechanisch andersrum läuft -> True
INVERT_OUTPUT = False

# Winkel:
OPEN_ANGLE = 180
CLOSE_ANGLE = 0


# =========================
# FS helpers
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_images_sorted(folder: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)

def load_paths_additive(templates_dir: str, dataset_dir: str) -> Tuple[List[str], str]:
    tpl_paths = list_images_sorted(templates_dir)
    ds_paths = list_images_sorted(dataset_dir)
    all_paths = list(dict.fromkeys(tpl_paths + ds_paths))[:MAX_FILES_PER_BANK]
    source = f"templates({len(tpl_paths)}) + dataset({len(ds_paths)}) | total_used={len(all_paths)}"
    return all_paths, source


# =========================
# Camera helpers
# =========================
def get_backend_candidates():
    sys = platform.system().lower()
    if "windows" in sys:
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    if "darwin" in sys or "mac" in sys:
        return [cv2.CAP_AVFOUNDATION, None]
    return [cv2.CAP_V4L2, None]

def backend_name(be):
    if be is None:
        return "default"
    mapping = {
        cv2.CAP_DSHOW: "DSHOW",
        cv2.CAP_MSMF: "MSMF",
        cv2.CAP_AVFOUNDATION: "AVFOUNDATION",
        cv2.CAP_V4L2: "V4L2",
    }
    return mapping.get(be, str(be))

def try_open_camera(index: int, backend):
    cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap

def set_resolution(cap, w, h):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

def get_resolution(cap):
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def open_best_camera(preferred_index: int, max_index: int, target_w: int, target_h: int):
    for be in get_backend_candidates():
        cap = try_open_camera(preferred_index, be)
        if cap is not None:
            set_resolution(cap, target_w, target_h)
            return cap, preferred_index, be
    for i in range(max_index):
        for be in get_backend_candidates():
            cap = try_open_camera(i, be)
            if cap is not None:
                set_resolution(cap, target_w, target_h)
                return cap, i, be
    raise RuntimeError("No camera found. Increase MAX_CAMERA_INDEX_TO_SCAN or check permissions.")

def list_available_cameras(max_index: int) -> List[int]:
    available = []
    for i in range(max_index):
        for be in get_backend_candidates():
            cap = try_open_camera(i, be)
            if cap is not None:
                available.append(i)
                cap.release()
                break
    return available


# =========================
# Matching
# =========================
def preprocess(gray):
    if USE_EDGES:
        return cv2.Canny(gray, 60, 160)
    return cv2.GaussianBlur(gray, (3, 3), 0)

def crop_search_roi(frame_bgr):
    # full frame when ratio <= 0.0
    if SEARCH_ROI_X_START_RATIO <= 0.0:
        return frame_bgr, (0, 0)
    h, w = frame_bgr.shape[:2]
    x1 = int(w * SEARCH_ROI_X_START_RATIO)
    roi = frame_bgr[:, x1:w]
    return roi, (x1, 0)

def crop_exact(frame, x, y, w, h):
    H, W = frame.shape[:2]
    x1 = max(0, min(W, x))
    y1 = max(0, min(H, y))
    x2 = max(0, min(W, x + w))
    y2 = max(0, min(H, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def prep_frame_small(frame_bgr, downscale: float) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    src = preprocess(gray)
    src_small = cv2.resize(src, (0, 0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    return src_small

def match_map_prepped(src_small: np.ndarray, tmpl_small: np.ndarray) -> Optional[np.ndarray]:
    th, tw = tmpl_small.shape[:2]
    Hs, Ws = src_small.shape[:2]
    if th >= Hs or tw >= Ws:
        return None
    return cv2.matchTemplate(src_small, tmpl_small, cv2.TM_CCOEFF_NORMED)

def topk_from_map(score_map: np.ndarray, k: int, min_dist: int) -> List[Tuple[float, int, int]]:
    if score_map is None:
        return []
    m = score_map.copy()
    out = []
    for _ in range(k):
        _, max_val, _, max_loc = cv2.minMaxLoc(m)
        if not np.isfinite(max_val):
            break
        x, y = max_loc
        out.append((float(max_val), x, y))
        x1 = max(0, x - min_dist)
        y1 = max(0, y - min_dist)
        x2 = min(m.shape[1] - 1, x + min_dist)
        y2 = min(m.shape[0] - 1, y + min_dist)
        m[y1:y2+1, x1:x2+1] = -1.0
    return out

class TemplateBank:
    def __init__(self, templates_dir: str, dataset_dir: str, tag: str):
        self.templates_dir = templates_dir
        self.dataset_dir = dataset_dir
        self.tag = tag
        self.items: List[Dict] = []
        self.source = "none"

    def rebuild(self, downscale: float):
        paths, source = load_paths_additive(self.templates_dir, self.dataset_dir)
        items = []
        variant_count = 0

        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue

            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            g = preprocess(g)
            h0, w0 = g.shape[:2]

            for s in SCALES:
                if variant_count >= MAX_VARIANTS_PER_BANK:
                    break

                w = max(8, int(w0 * s))
                h = max(8, int(h0 * s))
                tmpl = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA) if (w != w0 or h != h0) else g

                ws = max(8, int(w * downscale))
                hs = max(8, int(h * downscale))
                tmpl_small = cv2.resize(tmpl, (ws, hs), interpolation=cv2.INTER_AREA) if (ws != w or hs != h) else tmpl

                items.append({
                    "path": p,
                    "tmpl_small": tmpl_small,
                    "w_full": w,
                    "h_full": h,
                })
                variant_count += 1

            if variant_count >= MAX_VARIANTS_PER_BANK:
                break

        self.items = items
        self.source = source

def best_pos_candidates(frame_bgr, pos_bank: TemplateBank, downscale: float):
    src_small = prep_frame_small(frame_bgr, downscale)

    candidates = []
    for it in pos_bank.items:
        score_map = match_map_prepped(src_small, it["tmpl_small"])
        if score_map is None:
            continue
        peaks = topk_from_map(score_map, k=2, min_dist=MIN_CANDIDATE_DIST)
        for sc, x_s, y_s in peaks:
            x_full = int(x_s / downscale)
            y_full = int(y_s / downscale)
            w_full = int(it["w_full"])
            h_full = int(it["h_full"])
            candidates.append((float(sc), x_full, y_full, w_full, h_full, it["path"]))

    candidates.sort(key=lambda t: t[0], reverse=True)

    chosen = []
    for c in candidates:
        if len(chosen) >= TOPK_POS:
            break
        sc, x, y, w, h, p = c
        ok = True
        cx, cy = x + w // 2, y + h // 2
        for _, x2, y2, w2, h2, _ in chosen:
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
            if abs(cx - cx2) + abs(cy - cy2) < 80:
                ok = False
                break
        if ok:
            chosen.append(c)

    return chosen

def best_neg_score_in_crop(crop_bgr, neg_bank: TemplateBank, downscale: float) -> float:
    if not neg_bank.items:
        return 0.0

    src_small = prep_frame_small(crop_bgr, downscale)

    best = 0.0
    for it in neg_bank.items:
        score_map = match_map_prepped(src_small, it["tmpl_small"])
        if score_map is None:
            continue
        _, max_val, _, _ = cv2.minMaxLoc(score_map)
        best = max(best, float(max_val))
    return best

def detect_one_object(frame_bgr, pos_bank: TemplateBank, neg_bank: TemplateBank,
                      pos_thr: float, margin: float, downscale: float):
    cands = best_pos_candidates(frame_bgr, pos_bank, downscale)

    accepted = None
    best_dbg = (0.0, 0.0, 0.0)
    for pos_raw, x, y, w, h, pos_path in cands:
        if pos_raw < pos_thr:
            continue
        crop = crop_exact(frame_bgr, x, y, w, h)
        if crop is None:
            continue
        neg_local = best_neg_score_in_crop(crop, neg_bank, downscale)
        diff = pos_raw - neg_local
        if diff >= margin:
            if accepted is None or diff > best_dbg[2] or (abs(diff - best_dbg[2]) < 1e-6 and pos_raw > best_dbg[0]):
                accepted = (pos_raw, neg_local, x, y, w, h, pos_path)
                best_dbg = (pos_raw, neg_local, diff)

    dbg_pos = float(cands[0][0]) if len(cands) > 0 else 0.0
    return accepted, (best_dbg if accepted is not None else (dbg_pos, 0.0, 0.0))


# =========================
# Geometry helpers
# =========================
def rect_intersection_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> int:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return int((ix2 - ix1) * (iy2 - iy1))

def point_in_rect(px, py, x1, y1, x2, y2) -> bool:
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


# =========================
# Robot commander (UART like HandOnly)
# =========================
class RobotCommander:
    def __init__(self):
        if UART_AVAILABLE:
            try:
                u = uart_mod.uart()
                u.p0, u.p1, u.p2, u.p3, u.p4 = 10, 20, 30, 40, 50
                u.send()
                print("[UART TEST] OK: 10 20 30 40 50 gesendet")
            except Exception as e:
                print("\n[UART TEST] FEHLER:")
                print(repr(e))
                print()

    def _clamp(self, x: int) -> int:
        return int(max(SERVO_MIN, min(SERVO_MAX, int(x))))

    def _map_for_hw(self, x: int) -> int:
        x = self._clamp(x)
        if INVERT_OUTPUT:
            return SERVO_MAX - x
        return x

    def send_five(self, angles: List[int]):
        # angles = logische Winkel (0=zu,180=auf)
        logical = [self._clamp(a) for a in angles]
        hw = [self._map_for_hw(a) for a in logical]

        if not UART_AVAILABLE:
            print("[TX/NO-UART] logical:", logical, "| hw:", hw)
            return

        try:
            u = uart_mod.uart()
            thumb, index, middle, ring, pinky = hw
            u.p0 = int(pinky)
            u.p1 = int(ring)
            u.p2 = int(middle)
            u.p3 = int(index)
            u.p4 = int(thumb)
            u.send()
            print("[TX] logical:", logical, "| hw:", hw)
        except Exception as e:
            print("\n[UART SEND ERROR]")
            print(repr(e))
            print()


# =========================
# Threads
# =========================
class CameraStream:
    def __init__(self, preferred_idx: int):
        self.lock = threading.Lock()
        self.cap = None
        self.cam_idx = None
        self.backend = None
        self.running = False
        self.frame = None

        cap, idx, be = open_best_camera(preferred_idx, MAX_CAMERA_INDEX_TO_SCAN, TARGET_W, TARGET_H)
        self.cap = cap
        self.cam_idx = idx
        self.backend = be
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def start(self):
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def stop(self):
        self.running = False
        if hasattr(self, "th"):
            self.th.join(timeout=1.0)
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def switch(self, new_index: int) -> bool:
        with self.lock:
            for be in get_backend_candidates():
                newcap = try_open_camera(new_index, be)
                if newcap is not None:
                    set_resolution(newcap, TARGET_W, TARGET_H)
                    try:
                        newcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    if self.cap is not None:
                        self.cap.release()
                    self.cap = newcap
                    self.cam_idx = new_index
                    self.backend = be
                    self.frame = None
                    return True
        return False

    def get_info(self):
        with self.lock:
            if self.cap is None:
                return None
            aw, ah = get_resolution(self.cap)
            return self.cam_idx, backend_name(self.backend), aw, ah

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def _loop(self):
        while self.running:
            with self.lock:
                cap = self.cap
            if cap is None:
                time.sleep(0.01)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            if FLIP:
                frame = cv2.flip(frame, 1)
            with self.lock:
                self.frame = frame


class DualDetector:
    def __init__(self, cam: CameraStream,
                 glass_pos: TemplateBank, glass_neg: TemplateBank,
                 arm_pos: TemplateBank, arm_neg: TemplateBank):
        self.cam = cam
        self.glass_pos = glass_pos
        self.glass_neg = glass_neg
        self.arm_pos = arm_pos
        self.arm_neg = arm_neg

        self.lock = threading.Lock()
        self.running = False

        self.last_glass = None
        self.last_arm = None
        self.last_frame = None

        self.dbg_glass = (0.0, 0.0, 0.0)
        self.dbg_arm = (0.0, 0.0, 0.0)

    def start(self):
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def stop(self):
        self.running = False
        if hasattr(self, "th"):
            self.th.join(timeout=1.0)

    def clear(self):
        with self.lock:
            self.last_glass = None
            self.last_arm = None
            self.last_frame = None
            self.dbg_glass = (0.0, 0.0, 0.0)
            self.dbg_arm = (0.0, 0.0, 0.0)

    def get_last(self):
        with self.lock:
            g = self.last_glass
            a = self.last_arm
            f = None if self.last_frame is None else self.last_frame.copy()
            dbg_g = self.dbg_glass
            dbg_a = self.dbg_arm
            return g, a, f, dbg_g, dbg_a

    def _loop(self):
        period = 1.0 / max(1, DETECT_HZ)
        while self.running:
            t0 = time.time()
            frame = self.cam.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            search_roi, off = crop_search_roi(frame)

            g_acc, g_dbg = detect_one_object(
                search_roi, self.glass_pos, self.glass_neg,
                GLASS_THRESHOLD, GLASS_SCORE_MARGIN, FRAME_DOWNSCALE
            )
            a_acc, a_dbg = detect_one_object(
                search_roi, self.arm_pos, self.arm_neg,
                ARM_THRESHOLD, ARM_SCORE_MARGIN, FRAME_DOWNSCALE
            )

            if g_acc is not None:
                pr, nr, x, y, w, h, p = g_acc
                g_acc = (pr, nr, x + off[0], y + off[1], w, h, p)
            if a_acc is not None:
                pr, nr, x, y, w, h, p = a_acc
                a_acc = (pr, nr, x + off[0], y + off[1], w, h, p)

            with self.lock:
                self.last_glass = g_acc
                self.last_arm = a_acc
                self.last_frame = frame
                self.dbg_glass = g_dbg
                self.dbg_arm = a_dbg

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))


# =========================
# Hand logic (LATCH + ONE-SHOT)
# =========================
class HandLogic:
    """
    DU WOLLTEST: "greifen" und "schließen" tauschen.

    Deshalb:
    - ROI ENTRY  -> GREIFEN/SCHLIESSEN (CLOSE)
    - OVERLAP    -> ÖFFNEN (OPEN) (nach Delay)
    """
    def __init__(self, commander: RobotCommander):
        self.cmd = commander

        self.open_angles = [OPEN_ANGLE] * 5
        self.close_angles = [CLOSE_ANGLE] * 5

        # Startzustand: OFFEN
        self.state = "OPEN"  # "OPEN" or "CLOSED"
        self.pending_open_at: Optional[float] = None

        self.prev_arm_in_roi = False
        self.prev_overlap = False

        # Start einmal OPEN senden
        self.cmd.send_five(self.open_angles)

    def update(self, frame_shape, glass_box, arm_box):
        H, W = frame_shape[:2]

        rx1 = int(W * ROI_REL_X)
        ry1 = int(H * ROI_REL_Y)
        rx2 = int(rx1 + W * ROI_REL_W)
        ry2 = int(ry1 + H * ROI_REL_H)

        now = time.time()

        # ---------- arm_in_roi ----------
        arm_in_roi = False
        if arm_box is not None:
            ax1, ay1, ax2, ay2 = arm_box
            acx = (ax1 + ax2) // 2
            acy = (ay1 + ay2) // 2
            arm_in_roi = point_in_rect(acx, acy, rx1, ry1, rx2, ry2)

        roi_enter = arm_in_roi and (not self.prev_arm_in_roi)

        # ---------- overlap ----------
        overlap = False
        if glass_box is not None and arm_box is not None:
            gx1, gy1, gx2, gy2 = glass_box
            ax1, ay1, ax2, ay2 = arm_box
            area = rect_intersection_area(gx1, gy1, gx2, gy2, ax1, ay1, ax2, ay2)
            overlap = area >= REQUIRE_MIN_INTERSECTION_AREA

        overlap_enter = overlap and (not self.prev_overlap)
        overlap_exit = (not overlap) and self.prev_overlap

        # =========================================
        # NEW STATE MACHINE (SWAPPED)
        # =========================================

        # 1) ROI entry -> CLOSE sofort (greifen)
        if roi_enter:
            self.pending_open_at = None
            if self.state != "CLOSED":
                self.state = "CLOSED"
                self.cmd.send_five(self.close_angles)

        # 2) Overlap -> OPEN nach Delay (loslassen)
        if overlap_enter:
            self.pending_open_at = now + CLOSE_DELAY_S

        if overlap_exit:
            self.pending_open_at = None

        if self.pending_open_at is not None:
            if (not overlap):
                self.pending_open_at = None
            elif now >= self.pending_open_at:
                self.pending_open_at = None
                if self.state != "OPEN":
                    self.state = "OPEN"
                    self.cmd.send_five(self.open_angles)

        self.prev_arm_in_roi = arm_in_roi
        self.prev_overlap = overlap

        return (rx1, ry1, rx2, ry2), arm_in_roi, overlap


# =========================
# Main
# =========================
def main():
    ensure_dir(GLASS_DATASET_DIR)
    ensure_dir(GLASS_TEMPLATES_DIR)
    ensure_dir(NEG_DATASET_DIR)
    ensure_dir(NEG_TEMPLATES_DIR)
    ensure_dir(ARM_DATASET_DIR)
    ensure_dir(ARM_TEMPLATES_DIR)

    cams = list_available_cameras(MAX_CAMERA_INDEX_TO_SCAN)
    print(f"[Camera] available_indices={cams}")
    print(f"[SERVO] 0=zu, 180=auf | INVERT_OUTPUT={INVERT_OUTPUT}")
    print("[LOGIC] ROI entry = CLOSE (greifen) | overlap = OPEN (nach Delay)")

    glass_pos = TemplateBank(GLASS_TEMPLATES_DIR, GLASS_DATASET_DIR, "GLASS_POS")
    glass_neg = TemplateBank(NEG_TEMPLATES_DIR, NEG_DATASET_DIR, "NEG")
    arm_pos = TemplateBank(ARM_TEMPLATES_DIR, ARM_DATASET_DIR, "ARM_POS")
    arm_neg = TemplateBank(NEG_TEMPLATES_DIR, NEG_DATASET_DIR, "NEG")

    glass_pos.rebuild(FRAME_DOWNSCALE)
    glass_neg.rebuild(FRAME_DOWNSCALE)
    arm_pos.rebuild(FRAME_DOWNSCALE)
    arm_neg.rebuild(FRAME_DOWNSCALE)

    print(f"[GLASS pos] {glass_pos.source} | variants={len(glass_pos.items)}")
    print(f"[ARM   pos] {arm_pos.source} | variants={len(arm_pos.items)}")
    print(f"[NEG bank ] {glass_neg.source} | variants={len(glass_neg.items)}")

    cam = CameraStream(PREFERRED_CAMERA_INDEX)
    cam.start()

    det = DualDetector(cam, glass_pos, glass_neg, arm_pos, arm_neg)
    det.start()

    commander = RobotCommander()
    logic = HandLogic(commander)

    print("Keys:")
    print("  0-9 = Kamera wechseln")
    print("  r   = reload templates/datasets")
    print("  q/ESC = quit")

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            vis = frame.copy()
            H, W = vis.shape[:2]

            info = cam.get_info()
            cam_idx, be_name, aw, ah = info if info else (-1, "?", 0, 0)

            glass_m, arm_m, _, dbg_g, dbg_a = det.get_last()

            glass_box = None
            arm_box = None

            if glass_m is not None:
                pr, nr, x, y, wbox, hbox, _ = glass_m
                x2 = min(W - 1, x + wbox)
                y2 = min(H - 1, y + hbox)
                glass_box = (x, y, x2, y2)
                cv2.rectangle(vis, (x, y), (x2, y2), (0, 255, 0), 3)
                cv2.putText(vis, "GLASS", (x, max(18, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if arm_m is not None:
                pr, nr, x, y, wbox, hbox, _ = arm_m
                x2 = min(W - 1, x + wbox)
                y2 = min(H - 1, y + hbox)
                arm_box = (x, y, x2, y2)
                cv2.rectangle(vis, (x, y), (x2, y2), (255, 0, 0), 3)
                cv2.putText(vis, "ARM", (x, max(18, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            roi_rect, arm_in_roi, overlap = logic.update(frame.shape, glass_box, arm_box)
            rx1, ry1, rx2, ry2 = roi_rect
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            cv2.putText(vis, "ROI (entry -> CLOSE/GRAB)", (rx1, max(18, ry1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            status = f"Cam:{cam_idx} {be_name} {aw}x{ah} | state={logic.state} | arm_in_roi={arm_in_roi} | overlap={overlap}"
            if logic.pending_open_at is not None:
                status += f" | open_in={max(0.0, logic.pending_open_at - time.time()):.2f}s"
            cv2.putText(vis, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.putText(vis, f"GLASS diff={dbg_g[2]:+.2f} | ARM diff={dbg_a[2]:+.2f}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.putText(vis, "0-9=cam  r=reload  q=quit", (10, H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if ord("0") <= key <= ord("9"):
                target = key - ord("0")
                if info and target != cam_idx:
                    ok = cam.switch(target)
                    det.clear()
                    print(f"[Camera] switch to {target}: {'OK' if ok else 'FAILED'}")

            if key == ord("r"):
                glass_pos.rebuild(FRAME_DOWNSCALE)
                glass_neg.rebuild(FRAME_DOWNSCALE)
                arm_pos.rebuild(FRAME_DOWNSCALE)
                arm_neg.rebuild(FRAME_DOWNSCALE)
                det.clear()
                print(f"[GLASS pos] reloaded: {glass_pos.source} variants={len(glass_pos.items)}")
                print(f"[ARM   pos] reloaded: {arm_pos.source} variants={len(arm_pos.items)}")
                print(f"[NEG bank ] reloaded: {glass_neg.source} variants={len(glass_neg.items)}")

    finally:
        det.stop()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
