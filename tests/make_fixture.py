"""
A small synthetic input for the tests, `make run` and the release review (app.yml `review`), laid out
as physio-import writes its `physio` port (conventions.md), with known stimulation trains:

  <out>/physio/   recordings.csv, channels.csv, events.tsv, sub-*/<recording_id>.nwb
    sub-H01_rec-grid      human, 300 s: ECG (500 Hz, R-peaks known, HR drops during each train in
                          proportion to the amplitude), BP (mmHg, 120/80), respiration (15/min), and a
                          TTL trigger line `stim` (2 kHz, high for each 250 us... rounded to one sample)
                          with 10 trains: amplitude 0.5-2.5 mA x 10 / 20 Hz, 10 s each, every 28 s.
                          The amplitudes come from params/stim_log.csv (row 7 says 10 Hz where the
                          line pulses at 20 Hz: a table vs channel disagreement).
    sub-R01_rec-trials    rat, trial by trial (NWB timestamps with gaps): 6 trials of 25 s, train at
                          +10 s for 5 s; ECG 1 kHz, no BP (the "missing signals" path); `stim` line at
                          10 kHz marking each pulse's start and end with one-sample blips (as dataset
                          349); per-trial parameters in intervals/stimulation_source (current_uA,
                          frequency_Hz, pulse_width_us, duration_s, polarity, stim_shape) with the
                          trial windows as start/stop times.
    sub-R02_rec-closedloop  rat, 200 s at 1 kHz: two envelope channels as dataset 655: StimAmp
                          (3.8 from 40 to 100 s, sampled only in 0.2 s pieces with gaps, as the
                          Spike2 export) and CtrlAmp (+15 at 20-60 s = stimulation, -12 at 120-180 s =
                          kilohertz block); folder perf-...-gain05.
    sub-H02_rec-markers   human, 120 s, ECG only; markers in events.tsv: "baseline" at 5 s,
                          "VNS on 1.0mA 20Hz 250us" at 30 s / "VNS off" at 60 s, "VNS on 2.0mA 20Hz
                          250us" at 80 s / "VNS off" at 110 s, and "door opened" (not stimulation).
    sub-H03_rec-monitor   human, 12 s: a stimulus current monitor (A, 20 kHz): 3 trains of 2 s at
                          30 Hz, biphasic cathodic-first pulses, 0.8 mA, 200 us per phase, 50 us gap.
    sub-H04_rec-1mA       human, 60 s, ECG only, no stimulation channel; source file
                          raw/session-1/1mA_20Hz_200us.edf (parameters only in the file name).
  <out>/params/   stim_log.csv (sub-H01) and stim-map.yaml (envelope signs of sub-R02)
  <out>/truth.json

The NWB files are written with pynwb when it is installed, else with h5py in the same layout
(the platform's review generator image has h5py but not pynwb).

Usage: python -m tests.make_fixture <out-dir> [--writer pynwb|h5py]
"""
from __future__ import annotations

import csv
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 42


# --------------------------------------------------------------------------- signals

def ecg(t: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """A simple ECG (V): a narrow R wave (8 ms) and a T wave at each beat."""
    x = np.zeros_like(t)
    for b in beats:
        i0, i1 = np.searchsorted(t, [b - 0.05, b + 0.4])
        tt = t[i0:i1] - b
        x[i0:i1] += 1e-3 * np.exp(-0.5 * (tt / 0.008) ** 2) + 2.5e-4 * np.exp(-0.5 * ((tt - 0.25) / 0.04) ** 2)
    return x


def beat_times(dur: float, hr_of_t, t0: float = 0.3) -> np.ndarray:
    out, t = [], t0
    while t < dur:
        out.append(t)
        t += 60.0 / hr_of_t(t)
    return np.array(out)


# --------------------------------------------------------------------------- NWB writers

class Rec:
    def __init__(self, rid, subject, species, latin, source_path, source_format):
        self.rid, self.subject, self.species, self.latin = rid, subject, species, latin
        self.source_path, self.source_format = source_path, source_format
        self.series = []        # (name, data, unit, rate or None, timestamps or None, description, conversion)
        self.intervals = {}     # name -> (description, {col: list}) with start_time, stop_time
        self.channels = []      # channels.csv rows

    def add(self, name, data, unit, rate=None, timestamps=None, typ="other", original=None, site="",
            calibrated=True, conversion=1.0):
        self.series.append((name, data, unit, rate, timestamps, f"{original or name} (typing: name)", conversion))
        self.channels.append({"recording_id": self.rid, "channel": name, "type": typ, "original_name": original or name,
                              "unit": unit, "sampling_rate_hz": rate or round(1 / np.median(np.diff(timestamps))),
                              "n_samples": len(data), "calibrated": "true" if calibrated else "false",
                              "scale": 1, "site": site, "typing": "name", "status": "good",
                              "status_description": ""})

    def duration(self) -> float:
        end = 0.0
        for _, d, _, rate, ts, _, _ in self.series:
            end = max(end, (ts[-1] + 1.0 / 1000) if ts is not None else len(d) / rate)
        return end


def write_pynwb(rec: Rec, path: Path) -> None:
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.epoch import TimeIntervals
    from pynwb.file import Subject

    nwb = NWBFile(session_description=f"synthetic {rec.rid}", identifier=rec.rid,
                  session_start_time=datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
                  subject=Subject(subject_id=rec.subject, species=rec.latin))
    for name, data, unit, rate, ts, desc, conv in rec.series:
        kw = dict(rate=float(rate), starting_time=0.0) if ts is None else dict(timestamps=H5DataIO(ts, compression="gzip"))
        nwb.add_acquisition(TimeSeries(name=name, data=H5DataIO(data, compression="gzip"), unit=unit,
                                       conversion=conv, description=desc, **kw))
    for name, (desc, cols) in rec.intervals.items():
        ti = TimeIntervals(name=name, description=desc)
        for c in cols:
            if c not in ("start_time", "stop_time"):
                ti.add_column(name=c, description=c)
        for i in range(len(cols["start_time"])):
            ti.add_row(**{c: v[i] for c, v in cols.items()})
        nwb.add_time_intervals(ti)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def write_h5py(rec: Rec, path: Path) -> None:
    """The same layout with h5py only (NWB 2.x group/attribute names; enough for readers of the layout)."""
    import h5py

    def attrs(obj, ntype, **kw):
        obj.attrs["neurodata_type"] = ntype
        obj.attrs["namespace"] = "core"
        obj.attrs["object_id"] = str(uuid.uuid4())
        for k, v in kw.items():
            obj.attrs[k] = v

    with h5py.File(path, "w") as f:
        attrs(f, "NWBFile", nwb_version="2.9.0")
        f["identifier"] = rec.rid
        f["session_description"] = f"synthetic {rec.rid}"
        f["session_start_time"] = "2026-01-01T09:00:00+00:00"
        f["timestamps_reference_time"] = "2026-01-01T09:00:00+00:00"
        f.create_dataset("file_create_date", data=[datetime.now(timezone.utc).isoformat()],
                         dtype=h5py.string_dtype())
        for g in ("acquisition", "analysis", "processing", "stimulus/presentation", "stimulus/templates", "intervals"):
            f.require_group(g)
        sub = f.require_group("general/subject")
        attrs(sub, "Subject")
        sub["subject_id"] = rec.subject
        sub["species"] = rec.latin
        for name, data, unit, rate, ts, desc, conv in rec.series:
            g = f["acquisition"].create_group(name)
            attrs(g, "TimeSeries", description=desc, comments="no comments")
            d = g.create_dataset("data", data=data, compression="gzip")
            d.attrs.update(unit=unit, conversion=conv, offset=0.0, resolution=-1.0)
            if ts is None:
                s = g.create_dataset("starting_time", data=0.0)
                s.attrs.update(rate=float(rate), unit="seconds")
            else:
                s = g.create_dataset("timestamps", data=ts, compression="gzip")
                s.attrs.update(interval=1, unit="seconds")
        for name, (desc, cols) in rec.intervals.items():
            g = f["intervals"].create_group(name)
            attrs(g, "TimeIntervals", description=desc, colnames=[c for c in cols])
            n = len(cols["start_time"])
            g.create_dataset("id", data=np.arange(n))
            for c, v in cols.items():
                if isinstance(v[0], str):
                    g.create_dataset(c, data=v, dtype=h5py.string_dtype())
                else:
                    g.create_dataset(c, data=np.asarray(v))


# --------------------------------------------------------------------------- recordings

def rec_grid(truth: dict, rng) -> Rec:
    """Human, continuous: ECG, BP, resp and a TTL trigger with 10 trains; HR drop grows with amplitude."""
    rid = "sub-H01_rec-grid"
    dur = 300.0
    amps = [0.5, 1.0, 1.5, 2.0, 2.5, 0.5, 1.0, 1.5, 2.0, 2.5]
    freqs = [10, 10, 10, 10, 10, 20, 20, 20, 20, 20]
    onsets = [20.0 + 28.0 * k for k in range(10)]
    tdur = 10.0

    def hr(t):
        for o, a in zip(onsets, amps):
            if o <= t < o + tdur:
                return 70.0 - 6.0 * a
        return 70.0

    fs = 500.0
    t = np.arange(int(dur * fs)) / fs
    beats = beat_times(dur, hr)
    r = Rec(rid, "H01", "human", "Homo sapiens", "raw/H01/grid.edf", "edf")
    r.add("ecg", (ecg(t, beats) + 2e-5 * rng.standard_normal(len(t))).astype(np.float32), "V", fs, typ="ecg",
          original="ECG")
    bp = 100 + 20 * np.sin(2 * np.pi * (70 / 60) * t)
    r.add("bp", bp.astype(np.float32), "mmHg", fs, typ="bp", original="ABP")
    r.add("resp", np.sin(2 * np.pi * 0.25 * t).astype(np.float32), "a.u.", fs, typ="resp", original="Resp")
    fst = 2000.0
    ts = np.arange(int(dur * fst)) / fst
    ttl = np.zeros(len(ts), dtype=np.int16)
    for o, f in zip(onsets, freqs):
        for k in range(int(tdur * f)):
            i = int(round((o + k / f) * fst))
            ttl[i] = 1           # 0.5 ms high (one sample at 2 kHz)
    r.add("stim", ttl * 5, "V", fst, typ="stim", original="TTL stim", site="cervical vagus left")
    truth[rid] = {"events": [{"onset": o, "duration": tdur, "amplitude": a, "amplitude_unit": "mA",
                              "frequency_hz": f, "n_pulses": int(tdur * f), "pulse_width_us": 250}
                             for o, a, f in zip(onsets, amps, freqs)],
                  "r_peaks_s": [round(float(b), 4) for b in beats], "species": "human",
                  "disagreement_row": 7}
    return r


def rec_trials(truth: dict) -> Rec:
    """Rat, trial by trial with gaps; blips at pulse start and end; parameters in stimulation_source."""
    rid = "sub-R01_rec-trials"
    curr = [100, 200, 400, 600, 200, 600]
    freq = [5.0, 5.0, 5.0, 5.0, 20.0, 20.0]
    starts = [0.0, 40.0, 80.0, 120.0, 160.0, 200.0]
    win, pre, tdur, pw = 25.0, 10.0, 5.0, 200
    fs, fst = 1000.0, 10000.0
    tt, tst, ecgs, stims = [], [], [], []
    beats_all = []
    for s, c, f in zip(starts, curr, freq):
        t = s + np.arange(int(win * fs)) / fs
        hr_fn = (lambda c_: (lambda x: 380.0 - (0.08 * c_ if pre <= (x - (x // 40) * 40) < pre + tdur else 0)))(c)
        b = beat_times(s + win, lambda x: hr_fn(x), t0=s + 0.05)
        beats_all.extend(b.tolist())
        tt.append(t)
        ecgs.append(ecg(t, b))
        t2 = s + np.arange(int(win * fst)) / fst
        st = np.full(len(t2), -1.0)
        for k in range(int(tdur * f)):
            i = int(round((s + pre + k / f - s) * fst))
            st[i] = 1.0
            st[i + int(round(pw * 1e-6 * fst))] = 1.0     # end-of-phase blip
        tst.append(t2)
        stims.append(st)
    r = Rec(rid, "R01", "rat", "Rattus norvegicus", "primary/sub-R01/subject-R01.hdf5", "hdf5")
    r.add("ecg", np.concatenate(ecgs).astype(np.float32), "V", timestamps=np.concatenate(tt), typ="ecg",
          original="ecg/raw_ecg")
    r.add("stim", (np.concatenate(stims) * 1000).astype(np.float32), "a.u.", timestamps=np.concatenate(tst),
          typ="stim", original="neural/stim_timing", calibrated=False)
    r.intervals["stimulation_source"] = (
        "per-trial stimulation parameters copied from the HDF5 trial group attributes",
        {"start_time": starts, "stop_time": [s + win for s in starts], "current_uA": curr, "frequency_Hz": freq,
         "pulse_width_us": [pw] * 6, "duration_s": [tdur] * 6, "polarity": ["cathodic"] * 6,
         "stim_shape": ["biphasicwithdeadzone"] * 6})
    truth[rid] = {"events": [{"onset": s + pre, "duration": tdur, "amplitude": c, "amplitude_unit": "uA",
                              "frequency_hz": f, "n_pulses": int(tdur * f), "pulse_width_us": pw,
                              "polarity": "cathodic", "waveform": "biphasic-interphase"}
                             for s, c, f in zip(starts, curr, freq)], "species": "rat",
                  "r_peaks_s": [round(float(b), 4) for b in beats_all]}
    return r


def rec_closedloop(truth: dict) -> Rec:
    rid = "sub-R02_rec-closedloop"
    fs, dur = 1000.0, 200.0
    t = np.arange(int(dur * fs)) / fs
    r = Rec(rid, "R02", "rat", "Rattus norvegicus",
            "primary/sub-R02/perf-R02-set01-trial02-gain05/Perf-R02-set01-trial02-gain05.smrx.csv", "csv")
    r.add("ecg", ecg(t, beat_times(dur, lambda x: 350.0)).astype(np.float32), "V", fs, typ="ecg", original="3 ECG")
    sa = np.full(len(t), np.nan)
    # 0.2 s pieces every 0.6 s from 40 to 100 s (gaps, as the Spike2 export)
    for o in np.arange(40.0, 100.0, 0.6):
        i = int(o * fs)
        sa[i:i + 200] = 3.8 + 0.005 * np.sin(np.arange(200))
    ca = np.full(len(t), np.nan)
    ca[int(20 * fs):int(60 * fs)] = 15.0
    ca[int(120 * fs):int(180 * fs)] = -12.0 + 0.5 * np.sin(2 * np.pi * t[int(120 * fs):int(180 * fs)] / 30)
    r.add("stim", sa.astype(np.float32), "a.u.", fs, typ="stim", original="4 StimAmp", calibrated=False)
    r.add("stim-2", ca.astype(np.float32), "a.u.", fs, typ="stim", original="5 CtrlAmp", calibrated=False)
    truth[rid] = {"events": [
        {"onset": 20.0, "duration": 40.0, "trial_type": "stimulation", "amplitude": 15.0, "channel": "stim-2",
         "condition": "gain05"},
        {"onset": 40.0, "duration": 59.6, "trial_type": "stimulation", "amplitude": 3.8, "channel": "stim",
         "condition": "gain05"},
        {"onset": 120.0, "duration": 60.0, "trial_type": "block", "amplitude": 12.0, "channel": "stim-2",
         "waveform": "kilohertz", "condition": "gain05"}], "species": "rat"}
    return r


def rec_markers(truth: dict, markers: list) -> Rec:
    rid = "sub-H02_rec-markers"
    fs, dur = 250.0, 120.0
    t = np.arange(int(dur * fs)) / fs
    r = Rec(rid, "H02", "human", "Homo sapiens", "raw/H02/markers.edf", "edf")
    r.add("ecg", ecg(t, beat_times(dur, lambda x: 65.0)).astype(np.float32), "V", fs, typ="ecg", original="EKG")
    for on, lab in ((5.0, "baseline"), (30.0, "VNS on 1.0mA 20Hz 250us"), (45.0, "door opened"),
                    (60.0, "VNS off"), (80.0, "VNS on 2.0mA 20Hz 250us"), (110.0, "VNS off")):
        markers.append({"recording_id": rid, "onset": on, "duration": "n/a", "trial_type": "marker",
                        "event_id": f"mk-{len(markers) + 1:04d}", "label": lab, "source": "marker"})
    truth[rid] = {"events": [{"onset": 5.0, "trial_type": "baseline"},
                             {"onset": 30.0, "duration": 30.0, "amplitude": 1.0, "amplitude_unit": "mA",
                              "frequency_hz": 20.0, "pulse_width_us": 250},
                             {"onset": 80.0, "duration": 30.0, "amplitude": 2.0, "amplitude_unit": "mA",
                              "frequency_hz": 20.0, "pulse_width_us": 250}], "species": "human"}
    return r


def rec_monitor(truth: dict) -> Rec:
    rid = "sub-H03_rec-monitor"
    fs, dur = 20000.0, 12.0
    n = int(dur * fs)
    x = np.zeros(n, dtype=np.float32)
    onsets = [1.0, 5.0, 9.0]
    amp, pw, gap, f, tdur = 0.8e-3, 200e-6, 50e-6, 30.0, 2.0
    w, g = int(round(pw * fs)), int(round(gap * fs))
    for o in onsets:
        for k in range(int(tdur * f)):
            i = int(round((o + k / f) * fs))
            x[i:i + w] = -amp
            x[i + w + g:i + 2 * w + g] = amp
    x += (2e-6 * np.random.default_rng(SEED).standard_normal(n)).astype(np.float32)
    r = Rec(rid, "H03", "human", "Homo sapiens", "raw/H03/monitor.abf", "abf")
    r.add("stim", x, "A", fs, typ="stim", original="Stim monitor (I)")
    truth[rid] = {"events": [{"onset": o, "duration": tdur, "amplitude": 0.8, "amplitude_unit": "mA",
                              "frequency_hz": f, "n_pulses": int(tdur * f), "pulse_width_us": 200,
                              "polarity": "cathodic", "waveform": "biphasic-interphase"} for o in onsets],
                  "species": "human"}
    return r


def rec_filename(truth: dict) -> Rec:
    rid = "sub-H04_rec-1mA"
    fs, dur = 250.0, 60.0
    t = np.arange(int(dur * fs)) / fs
    r = Rec(rid, "H04", "human", "Homo sapiens", "raw/session-1/1mA_20Hz_200us.edf", "edf")
    r.add("ecg", ecg(t, beat_times(dur, lambda x: 60.0)).astype(np.float32), "V", fs, typ="ecg", original="ECG")
    truth[rid] = {"events": [{"onset": 0.0, "trial_type": "condition", "amplitude": 1.0, "amplitude_unit": "mA",
                              "frequency_hz": 20.0, "pulse_width_us": 200}], "species": "human"}
    return r


STIM_LOG = [  # sub-H01: one row per train, onset rounded and late by 0.3 s as a hand-kept log would be
    ("sub-H01_rec-grid", 20.3, 0.5, 10, 250), ("sub-H01_rec-grid", 48.3, 1.0, 10, 250),
    ("sub-H01_rec-grid", 76.3, 1.5, 10, 250), ("sub-H01_rec-grid", 104.3, 2.0, 10, 250),
    ("sub-H01_rec-grid", 132.3, 2.5, 10, 250), ("sub-H01_rec-grid", 160.3, 0.5, 20, 250),
    ("sub-H01_rec-grid", 188.3, 1.0, 10, 250),   # row 7: the log says 10 Hz, the line pulses at 20 Hz
    ("sub-H01_rec-grid", 216.3, 1.5, 20, 250), ("sub-H01_rec-grid", 244.3, 2.0, 20, 250),
    ("sub-H01_rec-grid", 272.3, 2.5, 20, 250)]

STIM_MAP = """\
# How to read the stimulation channels of sub-R02 (closed-loop control as Discover dataset 655)
version: 1
channels:
  - match: CtrlAmp
    kind: envelope
    site: right cervical vagus (control electrode)
    positive: {trial_type: stimulation}
    negative: {trial_type: block, waveform: kilohertz}
  - match: StimAmp
    kind: envelope
    site: right cervical vagus (perturbation electrode)
"""


def main(out, writer: str | None = None) -> Path:
    out = Path(out)
    rng = np.random.default_rng(SEED)
    if writer is None:
        try:
            import pynwb  # noqa: F401
            writer = "pynwb"
        except ImportError:
            writer = "h5py"
    write = write_pynwb if writer == "pynwb" else write_h5py
    physio = out / "physio"
    params = out / "params"
    physio.mkdir(parents=True, exist_ok=True)
    params.mkdir(parents=True, exist_ok=True)
    truth: dict = {"writer": writer}
    markers: list = []
    recs = [rec_grid(truth, rng), rec_trials(truth), rec_closedloop(truth), rec_markers(truth, markers),
            rec_monitor(truth), rec_filename(truth)]
    rec_rows, ch_rows = [], []
    for r in recs:
        rel = Path(f"sub-{r.subject}") / f"{r.rid}.nwb"
        (physio / rel.parent).mkdir(parents=True, exist_ok=True)
        write(r, physio / rel)
        types = sorted({c["type"] for c in r.channels})
        rec_rows.append({"recording_id": r.rid, "subject": r.subject, "session": "", "label": r.rid.split("rec-")[1],
                         "path": str(rel), "species": r.species, "start_time": "2026-01-01T09:00:00",
                         "duration_s": round(r.duration(), 3), "recorded_s": round(r.duration(), 3),
                         "n_segments": 1, "channel_types": "|".join(types), "source_format": r.source_format,
                         "source_path": r.source_path, "metadata_source": "file", "age": "", "sex": ""})
        ch_rows.extend(r.channels)
    for name, rows in (("recordings.csv", rec_rows), ("channels.csv", ch_rows)):
        with open(physio / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    cols = ["recording_id", "onset", "duration", "trial_type", "event_id", "amplitude", "amplitude_unit",
            "pulse_width_us", "frequency_hz", "train_duration_s", "train_on_s", "train_off_s", "n_pulses", "polarity",
            "waveform", "site", "contact", "condition", "label", "source"]
    with open(physio / "events.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for m in markers:
            w.writerow([m.get(c, "n/a") for c in cols])
    (physio / "provenance.json").write_text(json.dumps({"app": "physio-import", "version": "synthetic"}))
    with open(params / "stim_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "onset_s", "Amplitude (mA)", "frequency_Hz", "pulse_width_us"])
        w.writerows(STIM_LOG)
    (params / "stim-map.yaml").write_text(STIM_MAP)
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)
    w = None
    if "--writer" in args:
        w = args[args.index("--writer") + 1]
    main(args[0], w)
    print(f"wrote {args[0]}")
