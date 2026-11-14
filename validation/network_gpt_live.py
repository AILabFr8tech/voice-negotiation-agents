"""Network round trip to the GPT-Live API's front-end server.

Opens `--attempts` TCP connections to `api.openai.com:443` and measures how long
each one takes to be established, which is one network round trip. It sends
nothing and does not use the key: it only opens and closes the connection. Appends
a row to `network_gpt_live.csv` in `--output`.

    python -m validation.network_gpt_live
"""

from __future__ import annotations

import argparse
import csv
import socket
import statistics
import time
from datetime import date
from pathlib import Path
from typing import Final

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]
DATA: Final[Path] = RAIZ / "docs" / "report" / "data"
HOST: Final[str] = "api.openai.com"
PORT: Final[int] = 443
COLUMNS: Final[list[str]] = ["date", "host", "ip", "attempts", "p50_ms", "min_ms", "max_ms"]


def round_trip(ip: str, attempts: int, pause_s: float = 0.2) -> list[float]:
    timings = []
    for _ in range(attempts):
        t = time.perf_counter()
        with socket.create_connection((ip, PORT), timeout=5):
            timings.append((time.perf_counter() - t) * 1000)
        time.sleep(pause_s)
    return timings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--attempts", type=int, default=15)
    ap.add_argument("--output", type=Path, default=DATA)
    args = ap.parse_args(argv)
    # The IP is resolved once so DNS is not measured.
    ip = socket.getaddrinfo(HOST, PORT, proto=socket.IPPROTO_TCP)[0][4][0]
    t = round_trip(ip, args.attempts)
    row = {"date": date.today().isoformat(), "host": HOST, "ip": ip, "attempts": len(t),
            "p50_ms": round(statistics.median(t), 1), "min_ms": round(min(t), 1), "max_ms": round(max(t), 1)}
    ruta = args.output / "network_gpt_live.csv"
    new = not ruta.is_file()
    with ruta.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
