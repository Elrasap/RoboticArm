import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
from right_hand_kinematics import RightHandAngleTracker
import uart_mod

SEND_MODE = "curl"
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


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
    # (damit du sofort weißt ob send grundsätzlich klappt)
    try:
        u = uart_mod.uart()
        u.p0, u.p1, u.p2, u.p3, u.p4 = 10, 20, 30, 40, 50
        u.send()
        print("[UART TEST] OK: 10 20 30 40 50 gesendet")

    except Exception as e:
        print("\n[UART TEST] FEHLER -> run/Port/Permission/Protokoll Problem, nicht HandTracking:")
        print(e)
        print()
        # Nicht abbrechen: du kannst trotzdem Kamera testen

    # ---------------------------------------------------------

    t = RightHandAngleTracker(camera_index=0, flip=True)
    frame_i = 0

    print(f"HandOnly: q/ESC zum Beenden (sendet {SEND_MODE} über uart_mod/run)")

    try:
        while True:
            frame = t.update(draw=True)
            if frame is None:
                break

            kin = t.get_last_kin()
            values = extract_five_values_from_kin(kin, SEND_MODE)

            if frame_i % 30 == 0:
                print(
                    "[DBG] kin:",
                    "OK" if kin is not None else "None",
                    "| values:",
                    values if values is not None else "None",
                )

            if values is not None:
                ints = [int(round(v)) for v in values]

                try:
                    u = uart_mod.uart()  # neues Objekt
                    u.p0, u.p1, u.p2, u.p3, u.p4 = ints
                    u.send()

                    if frame_i % 30 == 0:
                        print("[TX]", ints)

                except Exception as e:
                    print("\n[UART SEND ERROR]")
                    print(e)
                    print()

            cv2.imshow("HandOnly", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            frame_i += 1

    finally:
        try:
            t.close()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
