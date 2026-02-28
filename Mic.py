import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import re
import sys
import json
import queue
import subprocess
from pathlib import Path

import sounddevice as sd
from vosk import Model, KaldiRecognizer

import uart_mod

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "vosk-model-small-de-0.15")
SAMPLE_RATE = 16000

# ============================================================
# SERVO RANGE + INVERT (FIX)
# ============================================================
# Du willst: 0 = geschlossen, 180 = offen
MIN_ANGLE = 0
MAX_ANGLE = 180

# Wenn es bei dir aktuell "verkehrt herum" läuft, dann invertieren.
# Dadurch wird 0 <-> 180 gedreht.
INVERT_OUTPUT = True

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# =========================
# PROCESS SCRIPTS
# =========================
CAM_SCRIPT = BASE_DIR / "Cam.py"
HAND_SCRIPT = BASE_DIR / "HandOnly.py"

cam_proc = None
hand_proc = None

def cam_is_active() -> bool:
    return cam_proc is not None and cam_proc.poll() is None

def hand_is_active() -> bool:
    return hand_proc is not None and hand_proc.poll() is None

def start_cam():
    global cam_proc
    if hand_is_active():
        print("[MIC] Hand aktiv -> erst 'hand aus'")
        return
    if cam_is_active():
        print("[MIC] Cam läuft schon")
        return
    cam_proc = subprocess.Popen([sys.executable, str(CAM_SCRIPT)], cwd=str(BASE_DIR))
    print("[MIC] Cam gestartet")

def stop_cam():
    global cam_proc
    if not cam_is_active():
        cam_proc = None
        print("[MIC] Cam war nicht aktiv")
        return
    cam_proc.terminate()
    try:
        cam_proc.wait(timeout=2.0)
    except Exception:
        cam_proc.kill()
    cam_proc = None
    print("[MIC] Cam gestoppt")

def start_hand():
    global hand_proc
    if cam_is_active():
        print("[MIC] Kamera aktiv -> erst 'kamera aus'")
        return
    if hand_is_active():
        print("[MIC] Hand läuft schon")
        return
    hand_proc = subprocess.Popen([sys.executable, str(HAND_SCRIPT)], cwd=str(BASE_DIR))
    print("[MIC] Hand gestartet (HandOnly.py)")

def stop_hand():
    global hand_proc
    if not hand_is_active():
        hand_proc = None
        print("[MIC] Hand war nicht aktiv")
        return
    hand_proc.terminate()
    try:
        hand_proc.wait(timeout=2.0)
    except Exception:
        hand_proc.kill()
    hand_proc = None
    print("[MIC] Hand gestoppt")

# =========================
# UART (HandOnly-Approach)
# =========================
def clamp_angle(x: int) -> int:
    if x < MIN_ANGLE:
        return MIN_ANGLE
    if x > MAX_ANGLE:
        return MAX_ANGLE
    return x

def maybe_invert(x: int) -> int:
    """0..180 invertieren, falls nötig."""
    x = clamp_angle(int(x))
    if INVERT_OUTPUT:
        return MAX_ANGLE - x
    return x

def uart_send_all(angles_dict):
    """
    sendet alle 5 Finger wie HandOnly:
      p0=pinky, p1=ring, p2=middle, p3=index, p4=thumb
    """
    # vals = [thumb,index,middle,ring,pinky]
    vals = [maybe_invert(angles_dict[f]) for f in FINGER_ORDER]
    thumb, index, middle, ring, pinky = vals

    msg_dbg = ",".join(str(v) for v in vals)
    print("[UART] would send (ALL):", msg_dbg)

    try:
        u = uart_mod.uart()
        u.p0 = int(pinky)
        u.p1 = int(ring)
        u.p2 = int(middle)
        u.p3 = int(index)
        u.p4 = int(thumb)
        u.send()
        print("[UART] gesendet (ALL):", msg_dbg)
    except Exception as e:
        print("[UART] send error:", e)

def uart_send_single(angles_dict, finger: str, angle: int):
    """
    Handshake-kompatibel bleiben: wir schicken "single finger" als kompletten 5er-Frame,
    nur eben mit dem einen Finger geändert (so ist der Microcontroller-Approach konsistent).
    """
    angles_dict[finger] = clamp_angle(int(angle))
    uart_send_all(angles_dict)

# =========================
# NUMBER PARSER (DE) - wie vorher
# =========================
def parse_number_from_text(text: str):
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        return int(m.group(1))

    t_raw = text.lower()

    t_raw = t_raw.replace("kleiner finger", " ")
    t_raw = t_raw.replace("kleinerfinger", " ")
    t_raw = re.sub(
        r"\b(daumen|baumeln|zeigefinger|mittelfinger|ringfinger|pinky|hand|ganze|finger|auf|zu|grad|winkel|kamera|an|aus)\b",
        " ",
        t_raw
    )

    t_raw = t_raw.replace("hunderteins", "hundert eins")
    t_raw = t_raw.replace("hundert eins", "hundert eins")

    parts = [p for p in re.split(r"\s+", t_raw.replace("-", " ").strip()) if p]
    if not parts:
        return None

    ones = {
        "null": 0, "ein": 1, "eins": 1, "eine": 1,
        "zwei": 2, "drei": 3, "vier": 4,
        "fünf": 5, "fuenf": 5, "sechs": 6,
        "sieben": 7, "acht": 8, "neun": 9
    }
    teens = {
        "zehn": 10, "elf": 11, "zwölf": 12, "zwoelf": 12,
        "dreizehn": 13, "vierzehn": 14,
        "fünfzehn": 15, "fuenfzehn": 15,
        "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19
    }
    tens = {
        "zwanzig": 20, "dreißig": 30, "dreissig": 30,
        "vierzig": 40, "fünfzig": 50, "fuenfzig": 50,
        "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90
    }

    def parse_compact(c: str):
        c = c.replace(" ", "").replace("-", "")

        if c in ones: return ones[c]
        if c in teens: return teens[c]
        if c in tens: return tens[c]

        if "hundert" in c:
            left, right = c.split("hundert", 1)
            base = 1
            if left:
                left = "ein" if left == "eins" else left
                base = ones.get(left, 1)
            rest = parse_compact(right) if right else 0
            return base * 100 + (rest or 0)

        m2 = re.match(
            r"^(ein|eins|zwei|drei|vier|fünf|fuenf|sechs|sieben|acht|neun)und"
            r"(zwanzig|dreißig|dreissig|vierzig|fünfzig|fuenfzig|sechzig|siebzig|achtzig|neunzig)$",
            c
        )
        if m2:
            a, b = m2.group(1), m2.group(2)
            a = "ein" if a in ("ein", "eins") else a
            return ones[a] + tens[b]

        return None

    candidates = []
    candidates.append(parts[-1])
    candidates.append("".join(parts))

    for k in range(2, min(7, len(parts)) + 1):
        candidates.append("".join(parts[-k:]))

    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for cand in candidates:
        v = parse_compact(cand)
        if v is not None:
            return v

    if len(parts) >= 3:
        for i in range(len(parts) - 2):
            a, u, b = parts[i], parts[i+1], parts[i+2]
            if u == "und" and a in ones and b in tens:
                return ones[a] + tens[b]

    return None

# =========================
# TEXT / FINGER
# =========================
def normalize(text: str) -> str:
    t = text.lower().strip()

    # häufige Vosk-Varianten vereinheitlichen
    t = t.replace("kleinerfinger", "kleiner finger")
    t = t.replace("kleinen finger", "kleiner finger")
    t = t.replace("kleine finger", "kleiner finger")
    t = t.replace("klein finger", "kleiner finger")

    t = t.replace("zeige finger", "zeigefinger")
    t = t.replace("ring finger", "ringfinger")
    t = t.replace("mittel finger", "mittelfinger")

    # pinky-Varianten
    t = t.replace("pinki", "pinky")
    t = t.replace("pinkie", "pinky")

    # grad-varianten
    t = t.replace("graut", "grad").replace("grat", "grad")

    # dein alter Spezialfall
    t = t.replace("warum in", "daumen")

    return t

def finger_from_text(text: str):
    t = text.lower()

    if "daumen" in t or "baumeln" in t:
        return "thumb"
    if "zeigefinger" in t:
        return "index"
    if "mittelfinger" in t:
        return "middle"
    if "ringfinger" in t:
        return "ring"

    # pinky: robust (auch wenn nur "klein" und "finger" irgendwo vorkommen)
    if "pinky" in t or "kleiner finger" in t or ("klein" in t and "finger" in t):
        return "pinky"

    return None

# =========================
# VOSK SETUP
# =========================
q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def main():
    # ---- UART quick self-test ----
    try:
        u = uart_mod.uart()
        u.p0, u.p1, u.p2, u.p3, u.p4 = 10, 20, 30, 40, 50
        u.send()
        print("[UART TEST] OK: 10 20 30 40 50 gesendet")
    except Exception as e:
        print("\n[UART TEST] FEHLER -> Port/Permission/Protokoll Problem:")
        print(e)
        print()

    print("[MIC] loading model:", MODEL_PATH)
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    # Startzustand: OFFEN (logisch 180). Wir senden aber ggf. invertiert an die Hardware.
    angles = {f: 180 for f in FINGER_ORDER}
    uart_send_all(angles)

    print("[MIC] ready.")
    print("Sag z.B.: 'Daumen 0 Grad' bis 'Daumen 180 Grad' (oder als Wort, z.B. 'null', 'einundneunzig')")
    print("Oder: 'ganze hand zu' -> alles 0 | 'ganze hand auf' -> alles 180")
    print("Zusätzlich: 'kamera an/aus' und 'hand an/aus' (exklusiv)")
    print("Stop: 'beenden' oder 'stopp'")
    print(f"[INFO] INVERT_OUTPUT={INVERT_OUTPUT} (dreht 0<->180 fürs Senden)")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = normalize(result.get("text", "").strip())
                if not text:
                    continue

                print("[MIC] erkannt:", text)

                if text in {"stopp", "stop", "beenden", "ende"}:
                    break

                # ===== MODE SWITCHES: CAM/HAND =====
                if text == "kamera an":
                    start_cam()
                    continue
                if text == "kamera aus":
                    stop_cam()
                    continue
                if text == "hand an":
                    start_hand()
                    continue
                if text == "hand aus":
                    stop_hand()
                    continue

                # Wenn Cam oder Hand aktiv ist: blockiere Servo-Kommandos
                if cam_is_active() or hand_is_active():
                    print("[MIC] Modus aktiv (Cam/Hand) -> ignoriere:", text)
                    continue

                # ===== HAND AUF/ZU (ohne "ganze") =====
                if ("hand" in text) and ("auf" in text) and (parse_number_from_text(text) is None):
                    for f in FINGER_ORDER:
                        angles[f] = 180
                    uart_send_all(angles)
                    continue

                if ("hand" in text) and ("zu" in text) and (parse_number_from_text(text) is None):
                    for f in FINGER_ORDER:
                        angles[f] = 0
                    uart_send_all(angles)
                    continue

                # Ganze Hand
                if "ganze hand" in text and "zu" in text:
                    for f in FINGER_ORDER:
                        angles[f] = 0
                    uart_send_all(angles)
                    continue

                if "ganze hand" in text and "auf" in text:
                    for f in FINGER_ORDER:
                        angles[f] = 180
                    uart_send_all(angles)
                    continue

                # Einzelner Finger + Zahl
                finger = finger_from_text(text)
                if finger is not None:
                    num = parse_number_from_text(text)
                    if num is None:
                        print("[MIC] Keine Zahl erkannt. Sag z.B. 'Daumen 120 Grad'")
                        continue
                    num = clamp_angle(int(num))
                    uart_send_single(angles, finger, num)
                    continue

                print("[MIC] Unbekanntes Kommando (Finger+Zahl | ganze hand auf/zu | kamera/hand an/aus)")

    # Cleanup: falls Mic beendet wird, Child-Prozesse stoppen
    stop_cam()
    stop_hand()
    print("[MIC] beendet.")

if __name__ == "__main__":
    main()
