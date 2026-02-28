import time
import serial
import serial.tools.list_ports

BAUDRATE = 115200

def pick_port():
    ports = list(serial.tools.list_ports.comports())
    print("Verfügbare Ports:")
    for p in ports:
        print(f"  - {p.device}: {p.description}")

    # Bevorzugt ACM/USB, aber Fibocom meiden
    candidates = []
    for p in ports:
        desc = (p.description or "").lower()
        dev = p.device.lower()
        score = 0
        if "fibocom" in desc:
            score -= 10
        if "ttyacm" in dev:
            score += 5
        if "ttyusb" in dev:
            score += 4
        if "arduino" in desc or "usb serial" in desc or "ch340" in desc or "cp210" in desc:
            score += 3
        candidates.append((score, p.device, p.description))

    candidates.sort(reverse=True, key=lambda x: x[0])
    if not candidates:
        raise RuntimeError("Keine Ports gefunden.")
    best = candidates[0]
    print(f"\nAuto-Auswahl: {best[1]} ({best[2]})")
    return best[1]

def safe_read_lines(ser, seconds=1.0):
    """Nicht-blockierend lesen, ohne readline(), um den Linux-Fehler zu umgehen."""
    buf = b""
    end = time.time() + seconds
    lines = []
    while time.time() < end:
        try:
            n = ser.in_waiting
            if n:
                buf += ser.read(n)
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    lines.append(line.decode("utf-8", errors="replace").strip())
            else:
                time.sleep(0.02)
        except serial.SerialException as e:
            print("[PY][ERR] SerialException beim Lesen:", repr(e))
            return lines, False
    return lines, True

def main():
    port = pick_port()
    print(f"\nÖffne {port} @ {BAUDRATE} ...")

    try:
        ser = serial.Serial(port, BAUDRATE, timeout=0)
    except Exception as e:
        print("FEHLER: Konnte Port nicht öffnen:", repr(e))
        print("Tipp: Port belegt? Rechte? (dialout) oder falscher Port.")
        return

    # Reset-Zeit
    time.sleep(2.0)

    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:
        pass

    print("Sende PING ...")
    try:
        ser.write(b"PING\n")
        ser.flush()
    except serial.SerialException as e:
        print("[PY][ERR] Schreiben fehlgeschlagen:", repr(e))
        print("→ Sehr wahrscheinlich Port wurde getrennt oder ist mehrfach geöffnet.")
        ser.close()
        return

    lines, ok = safe_read_lines(ser, seconds=1.5)
    for l in lines:
        print("[RX]", l)
    if not ok:
        print("\nWICHTIG: Lesen ist gecrasht → Port wurde getrennt ODER ein anderes Programm greift drauf zu.")
        print("Mach:  lsof /dev/ttyACM1  und stoppe ModemManager testweise.")
        ser.close()
        return

    print("\nSende CSV 5 Sekunden (10 Hz) ...")
    start = time.time()
    i = 0
    while time.time() - start < 5.0:
        msg = f"23,87,91,45,12,{i}\n".encode("utf-8")
        try:
            ser.write(msg)
            ser.flush()
            print("[TX]", msg.decode().strip())
        except serial.SerialException as e:
            print("[PY][ERR] SerialException beim Senden:", repr(e))
            break

        lines, ok = safe_read_lines(ser, seconds=0.2)
        for l in lines:
            print("[RX]", l)
        if not ok:
            break

        i += 1
        time.sleep(0.1)

    ser.close()
    print("Fertig.")

if __name__ == "__main__":
    main()
