import os
import subprocess


class uart:
    def __init__(self):
        # Instance-Variablen (nicht class vars)
        self.p0 = 1
        self.p1 = 1
        self.p2 = 1
        self.p3 = 1
        self.p4 = 1

        # run IMMER relativ zu diesem File finden (nicht zum aktuellen Working Dir)
        self._dir = os.path.dirname(os.path.abspath(__file__))
        self._run = os.path.join(self._dir, "run")

    def send(self):
        if not os.path.exists(self._run):
            raise FileNotFoundError(f"run nicht gefunden: {self._run}")

        cmd = [
            self._run,
            "-p", "0", str(self.p0),
            "-p", "1", str(self.p1),
            "-p", "2", str(self.p2),
            "-p", "3", str(self.p3),
            "-p", "4", str(180-self.p4),
        ]

        # WICHTIG: stderr/stdout einsammeln, damit du echte Fehler siehst
        r = subprocess.run(cmd, capture_output=True, text=True)

        if r.returncode != 0:
            raise RuntimeError(
                f"run failed (rc={r.returncode})\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{r.stdout}\n"
                f"STDERR:\n{r.stderr}\n"
            )

        # Optional: Ausgabe zeigen (hilft beim Debuggen enorm)
        if r.stderr.strip():
            print("[run stderr]", r.stderr.strip())
        if r.stdout.strip():
            print("[run stdout]", r.stdout.strip())

        return r
