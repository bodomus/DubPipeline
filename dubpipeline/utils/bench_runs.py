from __future__ import annotations

import argparse
import pathlib
import re
import statistics
import subprocess
import sys
from datetime import datetime


_RE_RUN_FINISHED = re.compile(r"\[TIMING\]\s+RUN finished in\s+(.+)$")


def _parse_duration_to_seconds(s: str) -> float:
    # форматы из ваших логов: "9m 09s", "11.2s", "1m 35s"
    s = s.strip()
    total = 0.0
    m = re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", s)
    for num, unit in m:
        v = float(num)
        if unit == "h":
            total += v * 3600
        elif unit == "m":
            total += v * 60
        elif unit == "s":
            total += v
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", type=str, default="bench_logs")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command after --")
    args = ap.parse_args()

    if not args.cmd or args.cmd[0] != "--":
        print("Usage: python tools/bench_runs.py --warmup 1 --runs 5 -- <your command...>")
        return 2

    cmd = args.cmd[1:]
    if cmd and cmd[0] == "-m":
        cmd = [sys.executable, *cmd]

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    times = []
    for i in range(args.warmup + args.runs):
        tag = "warmup" if i < args.warmup else f"run{i - args.warmup + 1:02d}"
        log_path = out_dir / f"{tag}.log.txt"

        print(f"[bench] {tag}: {' '.join(cmd)}")
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",)

        combined = (p.stdout or "") + "\n" + (p.stderr or "")
        log_path.write_text(combined, encoding="utf-8", errors="replace")

        # достаём итоговое время RUN из текста
        m = None
        for line in combined.splitlines()[::-1]:
            mm = _RE_RUN_FINISHED.search(line)
            if mm:
                m = mm
                break

        if p.returncode != 0:
            print(f"[bench] {tag}: FAILED (code={p.returncode}) log={log_path}")
            continue

        if m:
            sec = _parse_duration_to_seconds(m.group(1))
            if i >= args.warmup:
                times.append(sec)
            print(f"[bench] {tag}: RUN={m.group(1)} ({sec:.1f}s) log={log_path}")
        else:
            print(f"[bench] {tag}: could not parse RUN time, log={log_path}")

    if times:
        med = statistics.median(times)
        avg = sum(times) / len(times)
        print(f"\n[bench] runs={len(times)} median={med:.1f}s avg={avg:.1f}s min={min(times):.1f}s max={max(times):.1f}s")
    else:
        print("\n[bench] no successful runs to summarize")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
