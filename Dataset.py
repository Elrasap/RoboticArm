import os
import time
import glob
import platform
import threading
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

# =========================
# SETTINGS (Unified)
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

IMG_PREFIX = "img_"
IMG_EXT = ".jpg"
TPL_PREFIX = "tpl_"
TPL_EXT = ".jpg"

WINDOW_NAME = "DATASET (GLASS + ARM, NO YOLO) | green=glass, blue=arm"

PREFERRED_CAMERA_INDEX = 0
MAX_CAMERA_INDEX_TO_SCAN = 10
TARGET_W = 1280
TARGET_H = 720
FLIP = False

# Speed/quality settings
FRAME_DOWNSCALE = 0.35
SCALES = [1.0]
USE_EDGES = False

# ROI Search: rechte 60% (0.40) oder ganzes Bild (0.0)
SEARCH_ROI_X_START_RATIO = 0.40

# Detection loop frequency
DETECT_HZ = 15

MAX_FILES_PER_BANK = 10000
MAX_VARIANTS_PER_BANK = 20000

# --- glass thresholds
GLASS_THRESHOLD = 0.55
GLASS_SCORE_MARGIN = 0.08

# --- arm thresholds (oft etwas anders, bei Bedarf anpassen)
ARM_THRESHOLD = 0.55
ARM_SCORE_MARGIN = 0.08

TOPK_POS = 10
MIN_CANDIDATE_DIST = 40

BURST_N = 30
BURST_DELAY_S = 0.05


# =========================
# FS helpers
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def next_index(folder: str, prefix: str, ext: str) -> int:
    files = glob.glob(os.path.join(folder, f"{prefix}*{ext}"))
    max_i = -1
    for f in files:
        base = os.path.basename(f)
        num_part = base[len(prefix):-len(ext)]
        try:
            max_i = max(max_i, int(num_part))
        except ValueError:
            pass
    return max_i + 1

def save_image(folder: str, img, idx: int, prefix: str, ext: str) -> str:
    path = os.path.join(folder, f"{prefix}{idx:06d}{ext}")
    cv2.imwrite(path, img)
    return path

def list_images_sorted(folder: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)


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

def list_available_cameras(max_index: int):
    available = []
    for i in range(max_index):
        for be in get_backend_candidates():
            cap = try_open_camera(i, be)
            if cap is not None:
                available.append(i)
                cap.release()
                break
    return available

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


# =========================
# Unified Matcher
# =========================
def preprocess(gray):
    if USE_EDGES:
        return cv2.Canny(gray, 60, 160)
    return cv2.GaussianBlur(gray, (3, 3), 0)

def crop_exact(frame, x, y, w, h):
    H, W = frame.shape[:2]
    x1 = max(0, min(W, x))
    y1 = max(0, min(H, y))
    x2 = max(0, min(W, x + w))
    y2 = max(0, min(H, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def crop_search_roi(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if SEARCH_ROI_X_START_RATIO <= 0.0:
        return frame_bgr, (0, 0)
    x1 = int(w * SEARCH_ROI_X_START_RATIO)
    roi = frame_bgr[:, x1:w]
    return roi, (x1, 0)

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

def load_paths_additive(templates_dir: str, dataset_dir: str) -> Tuple[List[str], str]:
    tpl_paths = list_images_sorted(templates_dir)
    ds_paths = list_images_sorted(dataset_dir)
    all_paths = list(dict.fromkeys(tpl_paths + ds_paths))[:MAX_FILES_PER_BANK]
    source = f"templates({len(tpl_paths)}) + dataset({len(ds_paths)}) | total_used={len(all_paths)}"
    return all_paths, source

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
    """
    returns: accepted or None
      accepted = (pos_raw, neg_local, x, y, w, h, pos_path)
    """
    cands = best_pos_candidates(frame_bgr, pos_bank, downscale)

    accepted = None
    best_dbg = (0.0, 0.0, 0.0)  # pos_raw, neg_local, diff
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
    """
    Detects GLASS + ARM independently on the same frame.
    """
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

        self.last_glass = None  # (pos_raw, neg_local, x,y,w,h, path)
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

            # back to full coords
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
# Main UI
# =========================
def main():
    ensure_dir(GLASS_DATASET_DIR)
    ensure_dir(GLASS_TEMPLATES_DIR)
    ensure_dir(NEG_DATASET_DIR)
    ensure_dir(NEG_TEMPLATES_DIR)
    ensure_dir(ARM_DATASET_DIR)
    ensure_dir(ARM_TEMPLATES_DIR)

    glass_img_idx = next_index(GLASS_DATASET_DIR, IMG_PREFIX, IMG_EXT)
    neg_img_idx = next_index(NEG_DATASET_DIR, IMG_PREFIX, IMG_EXT)
    arm_img_idx = next_index(ARM_DATASET_DIR, IMG_PREFIX, IMG_EXT)

    glass_tpl_idx = next_index(GLASS_TEMPLATES_DIR, TPL_PREFIX, TPL_EXT)
    neg_tpl_idx = next_index(NEG_TEMPLATES_DIR, TPL_PREFIX, TPL_EXT)
    arm_tpl_idx = next_index(ARM_TEMPLATES_DIR, TPL_PREFIX, TPL_EXT)

    cams = list_available_cameras(MAX_CAMERA_INDEX_TO_SCAN)
    print(f"[Camera] available_indices={cams}")

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

    print("Keys:")
    print("  t = add GLASS template ROI           | c = save GLASS crop (green box)     | b = burst GLASS crops")
    print("  u = add ARM template ROI             | i = save ARM crop (blue box)       | o = burst ARM crops")
    print("  y = add NEG template ROI             | n = save NEG crop (manual ROI)")
    print("  r = reload banks")
    print("  0-9 switch cam | q/ESC quit")

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            vis = frame.copy()
            H, W = vis.shape[:2]

            # Search ROI anzeigen
            if SEARCH_ROI_X_START_RATIO > 0.0:
                sx1 = int(W * SEARCH_ROI_X_START_RATIO)
                cv2.rectangle(vis, (sx1, 0), (W - 1, H - 1), (200, 200, 200), 1)

            info = cam.get_info()
            cam_idx, be_name, aw, ah = info if info else (-1, "?", 0, 0)

            glass_m, arm_m, _, dbg_g, dbg_a = det.get_last()

            cv2.putText(vis, f"Cam:{cam_idx} {be_name} {aw}x{ah} | downscale={FRAME_DOWNSCALE} DET_HZ={DETECT_HZ}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

            cv2.putText(vis, f"GLASS thr={GLASS_THRESHOLD:.2f} margin={GLASS_SCORE_MARGIN:.2f} | pos={dbg_g[0]:.2f} neg={dbg_g[1]:.2f} diff={dbg_g[2]:+.2f}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            cv2.putText(vis, f"ARM   thr={ARM_THRESHOLD:.2f} margin={ARM_SCORE_MARGIN:.2f} | pos={dbg_a[0]:.2f} neg={dbg_a[1]:.2f} diff={dbg_a[2]:+.2f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

            cv2.putText(vis, "t/c/b=glass  u/i/o=arm  y/n=neg  r=reload  q=quit",
                        (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

            # Draw detections
            if glass_m is not None:
                pr, nr, x, y, wbox, hbox, _ = glass_m
                x2 = min(W - 1, x + wbox)
                y2 = min(H - 1, y + hbox)
                cv2.rectangle(vis, (x, y), (x2, y2), (0, 255, 0), 3)  # green
                cv2.putText(vis, "GLASS", (x, max(18, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if arm_m is not None:
                pr, nr, x, y, wbox, hbox, _ = arm_m
                x2 = min(W - 1, x + wbox)
                y2 = min(H - 1, y + hbox)
                cv2.rectangle(vis, (x, y), (x2, y2), (255, 0, 0), 3)  # blue (BGR)
                cv2.putText(vis, "ARM", (x, max(18, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

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

            # --------- GLASS template/crops ----------
            if key == ord("t"):
                det.clear()
                roi = cv2.selectROI("Select GLASS Template ROI", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select GLASS Template ROI")
                x, y, wroi, hroi = map(int, roi)
                if wroi > 5 and hroi > 5:
                    tpl = frame[y:y+hroi, x:x+wroi].copy()
                    path = save_image(GLASS_TEMPLATES_DIR, tpl, glass_tpl_idx, TPL_PREFIX, TPL_EXT)
                    glass_tpl_idx += 1
                    print(f"[GLASS tpl] saved: {path}")
                    glass_pos.rebuild(FRAME_DOWNSCALE)
                    det.clear()
                else:
                    print("[GLASS tpl] canceled/invalid ROI")

            if key == ord("c"):
                g2, a2, raw2, _, _ = det.get_last()
                if g2 is None or raw2 is None:
                    print("[Save GLASS] no accepted glass -> nothing saved")
                else:
                    _, _, x, y, wbox, hbox, *_ = g2
                    crop = crop_exact(raw2, x, y, wbox, hbox)
                    if crop is None:
                        print("[Save GLASS] crop invalid")
                    else:
                        path = save_image(GLASS_DATASET_DIR, crop, glass_img_idx, IMG_PREFIX, IMG_EXT)
                        glass_img_idx += 1
                        print(f"[Save GLASS] {path}")
                        glass_pos.rebuild(FRAME_DOWNSCALE)
                        det.clear()

            if key == ord("b"):
                saved = 0
                for _ in range(BURST_N):
                    g2, a2, raw2, _, _ = det.get_last()
                    if g2 is None or raw2 is None:
                        time.sleep(BURST_DELAY_S)
                        continue
                    _, _, x, y, wbox, hbox, *_ = g2
                    crop = crop_exact(raw2, x, y, wbox, hbox)
                    if crop is None:
                        time.sleep(BURST_DELAY_S)
                        continue
                    save_image(GLASS_DATASET_DIR, crop, glass_img_idx, IMG_PREFIX, IMG_EXT)
                    glass_img_idx += 1
                    saved += 1
                    time.sleep(BURST_DELAY_S)
                print(f"[Burst GLASS] saved {saved} crops")
                glass_pos.rebuild(FRAME_DOWNSCALE)
                det.clear()

            # --------- ARM template/crops ----------
            if key == ord("u"):
                det.clear()
                roi = cv2.selectROI("Select ARM Template ROI", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select ARM Template ROI")
                x, y, wroi, hroi = map(int, roi)
                if wroi > 5 and hroi > 5:
                    tpl = frame[y:y+hroi, x:x+wroi].copy()
                    path = save_image(ARM_TEMPLATES_DIR, tpl, arm_tpl_idx, TPL_PREFIX, TPL_EXT)
                    arm_tpl_idx += 1
                    print(f"[ARM tpl] saved: {path}")
                    arm_pos.rebuild(FRAME_DOWNSCALE)
                    det.clear()
                else:
                    print("[ARM tpl] canceled/invalid ROI")

            if key == ord("i"):
                g2, a2, raw2, _, _ = det.get_last()
                if a2 is None or raw2 is None:
                    print("[Save ARM] no accepted arm -> nothing saved")
                else:
                    _, _, x, y, wbox, hbox, *_ = a2
                    crop = crop_exact(raw2, x, y, wbox, hbox)
                    if crop is None:
                        print("[Save ARM] crop invalid")
                    else:
                        path = save_image(ARM_DATASET_DIR, crop, arm_img_idx, IMG_PREFIX, IMG_EXT)
                        arm_img_idx += 1
                        print(f"[Save ARM] {path}")
                        arm_pos.rebuild(FRAME_DOWNSCALE)
                        det.clear()

            if key == ord("o"):
                saved = 0
                for _ in range(BURST_N):
                    g2, a2, raw2, _, _ = det.get_last()
                    if a2 is None or raw2 is None:
                        time.sleep(BURST_DELAY_S)
                        continue
                    _, _, x, y, wbox, hbox, *_ = a2
                    crop = crop_exact(raw2, x, y, wbox, hbox)
                    if crop is None:
                        time.sleep(BURST_DELAY_S)
                        continue
                    save_image(ARM_DATASET_DIR, crop, arm_img_idx, IMG_PREFIX, IMG_EXT)
                    arm_img_idx += 1
                    saved += 1
                    time.sleep(BURST_DELAY_S)
                print(f"[Burst ARM] saved {saved} crops")
                arm_pos.rebuild(FRAME_DOWNSCALE)
                det.clear()

            # --------- NEG templates/crops ----------
            if key == ord("y"):
                det.clear()
                roi = cv2.selectROI("Select NEG Template ROI", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select NEG Template ROI")
                x, y, wroi, hroi = map(int, roi)
                if wroi > 5 and hroi > 5:
                    tpl = frame[y:y+hroi, x:x+wroi].copy()
                    path = save_image(NEG_TEMPLATES_DIR, tpl, neg_tpl_idx, TPL_PREFIX, TPL_EXT)
                    neg_tpl_idx += 1
                    print(f"[NEG tpl] saved: {path}")
                    glass_neg.rebuild(FRAME_DOWNSCALE)
                    arm_neg.rebuild(FRAME_DOWNSCALE)
                    det.clear()
                else:
                    print("[NEG tpl] canceled/invalid ROI")

            if key == ord("n"):
                det.clear()
                roi = cv2.selectROI("Select NEG CROP ROI", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select NEG CROP ROI")
                x, y, wroi, hroi = map(int, roi)
                if wroi > 5 and hroi > 5:
                    crop = frame[y:y+hroi, x:x+wroi].copy()
                    path = save_image(NEG_DATASET_DIR, crop, neg_img_idx, IMG_PREFIX, IMG_EXT)
                    neg_img_idx += 1
                    print(f"[Save NEG] {path}")
                    glass_neg.rebuild(FRAME_DOWNSCALE)
                    arm_neg.rebuild(FRAME_DOWNSCALE)
                    det.clear()
                else:
                    print("[Save NEG] canceled/invalid ROI")

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
