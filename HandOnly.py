import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import sys
from pathlib import Path
import time
import cv2

# ------------------------------------------------------------
# Fix: Script-Ordner ins PYTHONPATH, damit right_hand_kinematics gefunden wird
# ------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ------------------------------------------------------------
# Robust imports
# ------------------------------------------------------------
try:
    from right_hand_kinematics import RightHandAngleTracker
except Exception as e:
    print("\n[IMPORT ERROR] right_hand_kinematics.py konnte nicht importiert werden.")
    print("Check:")
    print(" - Liegen HandOnly.py und right_hand_kinematics.py im selben Ordner?")
    print(" - Startest du das Script ggf. aus einem anderen Working-Directory?")
    print("Fehler:", repr(e))
    raise

try:
    import uart_mod
    UART_AVAILABLE = True
except Exception as e:
    print("\n[WARN] uart_mod nicht verfügbar -> Script läuft ohne UART senden.")
    print("Fehler:", repr(e))
    UART_AVAILABLE = False

# Welche Werte vom Tracker benutzt werden sollen:
# "curl" (meist sinnvoll), oder "mcp"/"pip"/"dip"
SEND_MODE = "curl"

# Reihenfolge MUSS zu deinem Mapping passen:
# ints = [thumb,index,middle,ring,pinky]
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# ============================================================
# VOLLE SERVO AUSLENKUNG 0..180 + KORREKTE RICHTUNG
# ============================================================
SERVO_MIN = 0
SERVO_MAX = 180

# >>> WICHTIG: Damit "offen erkannt" auch wirklich öffnet <<<
# Wenn aktuell offen->schließt, dann muss das invertiert werden:
INVERT_OUTPUT = True

# Pro Finger beobachtete Min/Max-Werte (wird live gelernt)
_obs_min = {f: None for f in FINGER_ORDER}
_obs_max = {f: None for f in FINGER_ORDER}

def _clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def _update_minmax(finger: str, v: float):
    mn = _obs_min[finger]
    mx = _obs_max[finger]
    if mn is None or v < mn:
        _obs_min[finger] = v
    if mx is None or v > mx:
        _obs_max[finger] = v

def _apply_invert(out: int) -> int:
    out = int(_clamp(out, SERVO_MIN, SERVO_MAX))
    if INVERT_OUTPUT:
        return SERVO_MAX - out
    return out

def _remap_to_servo(finger: str, v: float) -> int:
    """
    Tracker-Wert v -> Servo 0..180, basierend auf beobachtetem min/max.
    Skaliert pro Finger dynamisch auf volle Range.
    """
    # Falls Tracker schon 0..180 liefert und noch kein min/max gelernt:
    if 0.0 <= v <= 180.0 and (_obs_min[finger] is None or _obs_max[finger] is None):
        out = int(round(_clamp(v, SERVO_MIN, SERVO_MAX)))
        return _apply_invert(out)

    mn = _obs_min[finger]
    mx = _obs_max[finger]

    # Wenn noch nicht genug Daten: direkt clampen (min/max wird trotzdem gelernt)
    if mn is None or mx is None:
        out = int(round(_clamp(v, SERVO_MIN, SERVO_MAX)))
        return _apply_invert(out)

    rng = mx - mn
    if rng < 1e-6:
        out = int(round(_clamp(v, SERVO_MIN, SERVO_MAX)))
        return _apply_invert(out)

    t = (v - mn) / rng
    t = _clamp(t, 0.0, 1.0)

    out_f = SERVO_MIN + t * (SERVO_MAX - SERVO_MIN)
    out = int(round(out_f))
    out = int(_clamp(out, SERVO_MIN, SERVO_MAX))
    return _apply_invert(out)

def extract_five_values_from_kin(kin, mode: str):
    if kin is None or kin.fingers is None:
        return None

    values = []
    for fn in FINGER_ORDER:
        f = kin.fingers.get(fn)
        if f is None:
            return None

        if mode == "mcp":
            v = f.mcp
        elif mode == "pip":
            v = f.pip
        elif mode == "dip":
            v = f.dip
        else:
            v = f.curl

        values.append(v)

    return values

def main():
    # ---- UART quick self-test ----
    if UART_AVAILABLE:
        try:
            u = uart_mod.uart()
            u.p0, u.p1, u.p2, u.p3, u.p4 = 10, 20, 30, 40, 50
            u.send()
            print("[UART TEST] OK: 10 20 30 40 50 gesendet")
        except Exception as e:
            print("\n[UART TEST] FEHLER -> Port/Permission/Protokoll Problem (Handtracking ist davon unabhängig):")
            print(repr(e))
            print()

    # flip=True = Selfie/mirror
    try:
        t = RightHandAngleTracker(camera_index=0, flip=True)
    except Exception as e:
        print("\n[START ERROR] Kamera/Tracker konnte nicht gestartet werden.")
        print("Check:")
        print(" - Kamera wird von anderer App genutzt?")
        print(" - Kamera index stimmt? (0/1/2...)")
        print("Fehler:", repr(e))
        raise

    last_send_time = 0.0
    SEND_INTERVAL = 0.333  # 10 Hz

    print("HandOnly läuft: q/ESC zum Beenden")
    print(f"Mode={SEND_MODE} | SERVO={SERVO_MIN}..{SERVO_MAX} | INVERT_OUTPUT={INVERT_OUTPUT}")

    try:
        while True:
            frame = t.update(draw=True)
            if frame is None:
                print("[INFO] Kein Frame von Kamera erhalten -> Ende.")
                break

            kin = t.get_last_kin()
            values = extract_five_values_from_kin(kin, SEND_MODE)

            now = time.time()

            # ---------- Senden ----------
            if values is not None and (now - last_send_time) >= SEND_INTERVAL:
                # 1) min/max pro Finger updaten (für dynamisches remap)
                for fn, v in zip(FINGER_ORDER, values):
                    try:
                        _update_minmax(fn, float(v))
                    except Exception:
                        pass

                # 2) remap auf volle Range 0..180
                mapped = []
                for fn, v in zip(FINGER_ORDER, values):
                    try:
                        mapped.append(_remap_to_servo(fn, float(v)))
                    except Exception:
                        mapped.append(_apply_invert(int(round(_clamp(float(v), SERVO_MIN, SERVO_MAX)))))

                # mapped ist [thumb,index,middle,ring,pinky]
                ints = mapped

                if UART_AVAILABLE:
                    try:
                        u = uart_mod.uart()
                        # Beibehaltung deiner Zuordnung (p0=pinky ... p4=thumb)
                        u.p0 = ints[4]  # pinky
                        u.p1 = ints[3]  # ring
                        u.p2 = ints[2]  # middle
                        u.p3 = ints[1]  # index
                        u.p4 = ints[0]  # thumb
                        u.send()
                        last_send_time = now
                        print("[TX]", ints, "| raw:", [int(round(x)) for x in values])
                    except Exception as e:
                        print("\n[UART SEND ERROR]")
                        print(repr(e))
                        print()
                else:
                    last_send_time = now
                    print("[TX/NO-UART]", ints, "| raw:", [int(round(x)) for x in values])

            cv2.imshow("HandOnly", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        try:
            t.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
