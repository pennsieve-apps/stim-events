"""
Compare stim-events' events.tsv for Discover dataset 349 (pig VNS grid) with the source file.

Truth per trial, from subject-<n>.hdf5: the group name is the stimulation onset (local wall clock),
the group attributes give current_uA, frequency_Hz, pulse_width_us, duration_s, polarity and
stim_shape. physio-import's recordings.csv start_time (same clock) places the onset in seconds.
The trigger-derived values (rate, pulse count, train duration, pulse width from the blips) are read
from the train the trigger channel gives, i.e. before the table overrides them: run stim-events a
second time on the physio folder with the NWB stimulation_source table hidden is not needed,
because qc.csv lists every table vs channel disagreement; none means they agree within tolerance.

Usage: python validation/validate_349.py <subject.hdf5> <physio dir> <stim-events dir>
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


def main(h5: str, physio: str, out: str) -> int:
    recs = list(csv.DictReader(open(Path(physio) / "recordings.csv")))
    ev = [r for r in csv.DictReader(open(Path(out) / "events.tsv"), delimiter="\t") if r["trial_type"] == "stimulation"]
    qc = list(csv.DictReader(open(Path(out) / "qc.csv")))
    t0 = datetime.fromisoformat(recs[0]["start_time"])
    truth = []
    with h5py.File(h5, "r") as f:
        for subj in f:
            for g in f[subj]:
                a = dict(f[subj][g].attrs)
                truth.append(((datetime.fromisoformat(g) - t0).total_seconds(), a))
    truth.sort(key=lambda x: x[0])
    print(f"trials in the file: {len(truth)}; stimulation rows: {len(ev)}")
    d_on, ok = [], 0
    for (on, a), r in zip(truth, ev):
        d_on.append(float(r["onset"]) - on)
        same = (float(r["amplitude"]) == a["current_uA"] and r["amplitude_unit"] == "uA"
                and float(r["frequency_hz"]) == a["frequency_Hz"] and float(r["pulse_width_us"]) == a["pulse_width_us"]
                and abs(float(r["train_duration_s"]) - a["duration_s"]) < 1e-6 and r["polarity"] == a["polarity"]
                and int(r["n_pulses"]) == int(round(a["frequency_Hz"] * a["duration_s"])))
        ok += same
    d_on = np.array(d_on)
    print(f"parameters equal to the attributes (and n_pulses = rate x duration): {ok}/{len(truth)}")
    print(f"onset - trial group time: median {np.median(d_on) * 1e3:.1f} ms, range {d_on.min() * 1e3:.1f} to "
          f"{d_on.max() * 1e3:.1f} ms")
    dis = [q for q in qc if q["check"] == "table_vs_channel"]
    print(f"table vs trigger disagreements (rate > 10 %, width > 20 %, duration > 0.5 s, pulses > 10 %): {len(dis)}")
    for q in dis:
        print("  ", q["note"])
    gaps = [q for q in qc if q["check"] == "baseline_gap"]
    print(f"trains with less than MIN_GAP_S before them: {len(gaps)}")
    return 0 if ok == len(truth) == len(ev) and not dis else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
