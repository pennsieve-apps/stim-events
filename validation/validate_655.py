"""
Check stim-events' envelope epochs for Discover dataset 655 against the protocol in its README.

Protocol (curator's notes of the dataset): the controller (CtrlAmp, control electrode) is switched
on; 2 minutes later the perturbation electrode (StimAmp) delivers a 10-minute 1 -> 30 -> 1 Hz ramp;
the controller stays on for 2 more minutes, then is turned off. Positive CtrlAmp = stimulation,
negative = kilohertz block. The controller gain is in the folder name (gain02 / gain05 / gain10).

Usage: python validation/validate_655.py <stim-events dir>
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def main(out: str) -> int:
    ev = list(csv.DictReader(open(Path(out) / "events.tsv"), delimiter="\t"))
    by = defaultdict(list)
    for r in ev:
        by[r["recording_id"]].append(r)
    print("recording | gain | controller on (s) | ramp start | ramp length (s) | ctrl on -> ramp (s) | "
          "ramp end -> ctrl off (s) | stim / block epochs | earlier perturbation-electrode epoch")
    ok = True
    for rid, rows in sorted(by.items()):
        pert = [r for r in rows if "perturbation" in r["site"]]
        ramp = sorted(pert, key=lambda r: float(r["duration"]))[-1:]
        first = [r for r in pert if r not in ramp and float(r["duration"]) > 10]
        ctrl = [r for r in rows if "control" in r["site"]]
        if len(ramp) != 1 or not ctrl:
            print(rid, "ramp or controller epochs not found")
            ok = False
            continue
        rp = ramp[0]
        r0, r1 = float(rp["onset"]), float(rp["onset"]) + float(rp["duration"])
        c0 = min(float(r["onset"]) for r in ctrl)
        c1 = max(float(r["onset"]) + float(r["duration"]) for r in ctrl)
        n_st = sum(r["trial_type"] == "stimulation" for r in ctrl)
        n_bl = sum(r["trial_type"] == "block" for r in ctrl)
        print(f"{rid} | {rp['condition'] if rp['condition'] != 'perturbation ramp' else ctrl[0]['condition']} | "
              f"{c0:.1f} | {r0:.1f} | {r1 - r0:.1f} | {r0 - c0:.1f} | {c1 - r1:.1f} | {n_st} / {n_bl}"
              + (f" | initial perturbation-electrode epoch {float(first[0]['onset']):.1f} s for "
                 f"{float(first[0]['duration']):.1f} s" if first else ""))
        ok &= abs((r1 - r0) - 600) < 10 and abs((r0 - c0) - 120) < 10
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
