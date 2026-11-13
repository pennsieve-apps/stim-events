"""
stim-events: stimulation epochs and their parameters, as one events.tsv.

  INPUT_DIR   physio/  physio-import's output (recordings.csv, channels.csv, events.tsv, NWB per
                       recording) or any NWB files
              params/  optional: parameter tables (CSV/TSV/XLSX), stim-map.yaml, SDS metadata
              (both are found by name and content, recursively, also when merged side by side)
  OUTPUT_DIR/stim-events/
      events.tsv  stim_summary.csv  qc.csv  features.csv  figures/<recording_id>_timeline.png
      summary.json  provenance.json

One recording at a time; a recording that fails is logged and skipped; a recording without any
stimulation source is listed as skipped (not a failure).
Exit codes: 0 ok, 1 no usable input / every recording failed, 2 bad parameters.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepare_environment() -> None:
    """Lambda-safe start: HOME and TMPDIR may be fresh, empty folders and everything else read-only."""
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    for var, default in (("TMPDIR", tmp), ("HOME", tmp / "home")):
        p = Path(os.environ.get(var) or default)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        os.environ[var] = str(p)
    mpl = Path(os.environ["TMPDIR"]) / "mpl"
    baked = Path(os.environ.get("STIM_EVENTS_MPL_CACHE", "/opt/stim-events/matplotlib"))
    if not os.environ.get("MPLCONFIGDIR") or not os.access(os.environ["MPLCONFIGDIR"], os.W_OK):
        mpl.mkdir(parents=True, exist_ok=True)
        if baked.is_dir():
            for f in baked.glob("fontlist-*.json"):
                target = mpl / f.name
                if not target.exists():
                    target.write_bytes(f.read_bytes())
        os.environ["MPLCONFIGDIR"] = str(mpl)
    os.environ.setdefault("MPLBACKEND", "Agg")


_prepare_environment()

from processor import resources as R  # noqa: E402

_requested = os.environ.get("THREADS", "0").strip() or "0"
try:
    _THREADS, _THREADS_WHY = R.resolve_workers(int(_requested), per_worker_mb=200, reserve_mb=400)
except ValueError:
    _THREADS, _THREADS_WHY = 1, f"THREADS={_requested!r} is not an integer"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(_THREADS)

import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from importlib import metadata as importlib_metadata  # noqa: E402

import numpy as np  # noqa: E402

from processor import detect as D  # noqa: E402
from processor import inputs as I  # noqa: E402
from processor import merge as M  # noqa: E402
from processor import params as P  # noqa: E402
from processor import stimmap as SM  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
log = logging.getLogger("stim-events")
log.setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

APP_NAME = "stim-events"
REPO_ROOT = Path(__file__).resolve().parent.parent
TRUE, FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")
CITATIONS = ["10.7554/eLife.78362", "10.1038/sdata.2016.44"]

DEFAULTS = {
    "TRAIN_GAP_S": "2", "PULSE_MERGE_S": "0.002", "TRIGGER_THRESHOLD": "auto", "ENVELOPE_THRESHOLD": "auto",
    "ENVELOPE_GAP_S": "5", "ENVELOPE_STEP": "0.25", "MIN_GAP_S": "10", "MATCH_TOLERANCE_S": "2",
    "MARKER_PATTERN": "auto", "NAME_PATTERN": "auto", "USE_SDS": "true", "FIGURES": "true", "THREADS": "0",
}

STIM_MARKER = re.compile(r"(?i)\b(stim\w*|vns|tens|tns|dbs|scs|train|pulses?|khfac|block|burst)\b")
BASELINE_MARKER = re.compile(r"(?i)\b(baseline|rest(ing)?|pre-?stim(ulus|ulation)?)\b")
ON_WORD = re.compile(r"(?i)\b(on|start|begin|onset)\b")
OFF_WORD = re.compile(r"(?i)\b(off|stop|end|offset)\b")

SUMMARY_HEADER = ["recording_id", "status", "n_stimulations", "n_blocks", "n_baseline", "n_conditions",
                  "parameters_varied", "amplitude_min", "amplitude_max", "amplitude_unit", "frequency_hz_min",
                  "frequency_hz_max", "pulse_width_us_min", "pulse_width_us_max", "train_duration_s_min",
                  "train_duration_s_max", "total_stimulation_s", "sources"]
QC_HEADER = ["recording_id", "check", "value", "note"]
FEATURE_HEADER = ["recording_id", "channel", "feature", "value"]
VARY_KEYS = ["amplitude", "frequency_hz", "pulse_width_us", "train_duration_s", "polarity", "waveform", "site",
             "contact", "condition"]


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "h5py", "openpyxl", "PyYAML", "matplotlib"):
        try:
            out[pkg.lower()] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def _bool(name: str, v: str) -> bool:
    s = str(v).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    raise ValueError(f"{name} must be true or false, got {v!r}")


def _pos(name: str, v: str, allow_zero: bool = False) -> float:
    try:
        f = float(v)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {v!r}") from None
    if not math.isfinite(f) or f < 0 or (f == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'>= 0' if allow_zero else '> 0'}, got {v!r}")
    return f


def _auto_num(name: str, v: str) -> float | None:
    s = str(v).strip().lower()
    if s in ("auto", ""):
        return None
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"{name} must be auto or a number, got {v!r}") from None


def _pattern(name: str, v: str):
    s = str(v).strip()
    if s.lower() in ("auto", "none", ""):
        return s.lower() or "none"
    try:
        rx = re.compile(s)
    except re.error as e:
        raise ValueError(f"{name} is not a valid regular expression: {e}") from None
    return rx


def read_params(env=os.environ) -> tuple[dict, dict]:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p = {
        "settings": D.Settings(
            train_gap_s=_pos("TRAIN_GAP_S", raw["TRAIN_GAP_S"]),
            pulse_merge_s=_pos("PULSE_MERGE_S", raw["PULSE_MERGE_S"], allow_zero=True),
            trigger_threshold=_auto_num("TRIGGER_THRESHOLD", raw["TRIGGER_THRESHOLD"]),
            envelope_threshold=_auto_num("ENVELOPE_THRESHOLD", raw["ENVELOPE_THRESHOLD"]),
            envelope_gap_s=_pos("ENVELOPE_GAP_S", raw["ENVELOPE_GAP_S"], allow_zero=True),
            envelope_step=_pos("ENVELOPE_STEP", raw["ENVELOPE_STEP"])),
        "min_gap_s": _pos("MIN_GAP_S", raw["MIN_GAP_S"], allow_zero=True),
        "tol": _pos("MATCH_TOLERANCE_S", raw["MATCH_TOLERANCE_S"], allow_zero=True),
        "marker_pattern": _pattern("MARKER_PATTERN", raw["MARKER_PATTERN"]),
        "name_pattern": _pattern("NAME_PATTERN", raw["NAME_PATTERN"]),
        "use_sds": _bool("USE_SDS", raw["USE_SDS"]),
        "figures": _bool("FIGURES", raw["FIGURES"]),
    }
    if p["settings"].pulse_merge_s >= p["settings"].train_gap_s:
        raise ValueError("PULSE_MERGE_S must be smaller than TRAIN_GAP_S")
    try:
        int(raw["THREADS"])
    except ValueError:
        raise ValueError("THREADS must be an integer") from None
    return raw, p


def num(x, nd: int = 6):
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return round(xf, nd) if math.isfinite(xf) else None


# --------------------------------------------------------------------------- parameter tables

class TableSet:
    """All parameter tables of the params port, with the rows of each assigned to recordings."""

    def __init__(self, paths: list[Path], root: Path, smap: SM.StimMap):
        self.tables = []   # (path, rule, rows, roles, has_time, recording columns)
        for path in paths:
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                rel = path.name
            rule = smap.table_rule(path, rel)
            try:
                rows = I.read_table(path, rule.sheet if rule else None)
            except Exception as e:  # noqa: BLE001
                log.warning("params: %s could not be read (%s)", rel, e)
                continue
            if not rows:
                continue
            cols = list(rows[0].keys())
            roles = {c: P.classify_column(c) for c in cols}
            if rule:
                for ours, theirs in rule.columns.items():
                    if theirs not in roles:
                        log.warning("params: %s has no column %r (stim-map.yaml)", rel, theirs)
                        continue
                    unit = P.split_unit(P.snake(theirs))[1]
                    if ours == "amplitude":
                        unit = unit if unit and unit not in ("s", "ms", "us", "Hz") else rule.amplitude_unit
                    roles[theirs] = P.ColumnRole(theirs, ours, unit, ours)
                if rule.recording_column:
                    if rule.recording_column in roles:
                        roles[rule.recording_column] = P.ColumnRole(rule.recording_column, "recording", None,
                                                                    "recording")
                    else:
                        log.warning("params: %s has no column %r (recording_column)", rel, rule.recording_column)
            useful = [r for r in roles.values() if r.role not in ("extra", "recording", "onset", "stop")]
            if not useful:
                log.info("params: %s has no stimulation parameter columns; ignored", rel)
                continue
            self.tables.append({"path": path, "rel": rel, "rule": rule, "rows": rows, "roles": roles,
                                "has_time": any(r.role == "onset" for r in roles.values()),
                                "rec_cols": [c for c, r in roles.items() if r.role == "recording"]})

    @staticmethod
    def _row_is_rec(row: dict, rec_cols: list[str], rec: I.Recording) -> bool:
        from fnmatch import fnmatch  # noqa: PLC0415
        src = rec.source_path.replace("\\", "/")
        parts = set(Path(src).parts) | {Path(src).stem, Path(src).name}
        subj = rec.subject.removeprefix("sub-")
        for c in rec_cols:
            v = row.get(c)
            if v is None or str(v).strip() == "":
                continue
            v = str(v).strip()
            vs = v.removeprefix("sub-")
            ok = (v == rec.recording_id or (subj and vs == subj) or v == rec.session
                  or v in parts or Path(v).stem in parts or (src and (src.endswith(v) or fnmatch(src, v)))
                  or (len(v) >= 4 and v in rec.recording_id))
            if not ok:
                return False
        return True

    def records_for(self, rec: I.Recording, n_recs: int) -> tuple[list[tuple[str, list[M.Record]]], list[str]]:
        """[(match mode, records)] per table for this recording, and notes."""
        out, notes = [], []
        for t in self.tables:
            rows = t["rows"]
            if t["rec_cols"]:
                rows = [r for r in rows if self._row_is_rec(r, t["rec_cols"], rec)]
            elif n_recs > 1:
                stem = t["path"].stem
                if not (stem in rec.recording_id or (rec.source_path and stem in rec.source_path)):
                    continue
            if not rows:
                continue
            rule = t["rule"]
            tu = rule.time_unit if rule else "s"
            recs = []
            for i, row in enumerate(rows):
                prm, ext = P.from_record(row, t["roles"], rule.amplitude_unit if rule else None)
                start = stop = None
                for c, role in t["roles"].items():
                    if role.role in ("onset", "stop"):
                        f = P.to_float(row.get(c))
                        if f is not None:
                            f = P.time_value(f, role.unit or tu, "s")
                            if role.role == "onset":
                                start = f
                            else:
                                stop = f
                recs.append(M.Record("table", start, stop, prm, ext, f"{t['rel']} row {i + 2}"))
            mode = (rule.match if rule and rule.match else None) or ("time" if t["has_time"] else "auto")
            out.append((mode, recs))
        return out, notes


# --------------------------------------------------------------------------- SDS metadata

def sds_layer(rec: I.Recording, sds: dict[str, Path]) -> tuple[dict, list[str]]:
    """Stimulation parameters from SDS manifest / samples / performances / subjects rows of this recording."""
    params: dict = {}
    used = []
    src = rec.source_path.replace("\\", "/")
    parts = set(Path(src).parts)
    for kind in ("manifest", "performances", "samples", "subjects"):
        path = sds.get(kind)
        if path is None:
            continue
        try:
            rows = I.read_table(path)
        except Exception as e:  # noqa: BLE001
            log.warning("SDS %s could not be read: %s", path.name, e)
            continue
        for row in rows:
            keyed = {P.snake(k): v for k, v in row.items()}
            hit = False
            if kind == "manifest":
                fn = str(keyed.get("filename") or "").strip().replace("\\", "/")
                hit = bool(fn) and (src.endswith(fn) or fn.endswith(src) or Path(fn).name == Path(src).name)
            else:
                idkey = {"performances": "performance_id", "samples": "sample_id", "subjects": "subject_id"}[kind]
                v = str(keyed.get(idkey) or "").strip()
                hit = bool(v) and (v in parts or any(p.startswith(v) for p in parts)
                                   or (kind == "subjects" and v.removeprefix("sub-") == rec.subject.removeprefix("sub-")))
            if not hit:
                continue
            used.append(f"{path.name}")
            for c, v in row.items():
                role = P.classify_column(c)
                if role.role in P.PARAMS and role.role not in ("label", "trial_type", "condition"):
                    P.set_param(params, role.role, v, role.unit)
                if role.role == "label" or P.snake(c) in ("additional_metadata", "description"):
                    if v:
                        for k, val in P.parse_tokens(str(v)).items():
                            params.setdefault(k, val)
    return params, sorted(set(used))


# --------------------------------------------------------------------------- one recording

def channel_layer(e: D.Epoch, kind: str, unit: str, rule: SM.ChannelRule | None, row: dict) -> tuple[dict, list[str]]:
    """events.tsv parameters of one channel epoch."""
    notes = []
    p: dict = {}
    scale = rule.scale if rule else 1.0
    if kind in ("trigger", "monitor"):
        if e.khz:
            p.update(trial_type="block", waveform="kilohertz", frequency_hz=e.frequency_hz,
                     train_duration_s=e.duration)
        else:
            p["trial_type"] = "stimulation"
            if e.frequency_hz:
                p["frequency_hz"] = e.frequency_hz
            if e.n_pulses:
                p["n_pulses"] = e.n_pulses
            if e.pulse_width_us is not None:
                p["pulse_width_us"] = e.pulse_width_us
            if e.n_pulses and e.n_pulses >= 2:
                p["train_duration_s"] = e.duration
        if kind == "monitor" and e.amplitude is not None:
            a, u = P.amplitude_value(e.amplitude * scale, unit)
            p["amplitude"], p["amplitude_unit"] = a, u
            if e.polarity:
                p["polarity"] = e.polarity
            if e.waveform and "waveform" not in p:
                p["waveform"] = e.waveform
    else:
        v = (e.amplitude or 0.0) * scale
        mapping = (rule.negative if e.sign < 0 else rule.positive) if rule else {}
        if e.sign < 0 and rule and rule.negative and rule.negative.get("absolute", True):
            v = abs(v)
        q = rule.quantity if rule else "amplitude"
        if q == "amplitude":
            a, u = P.amplitude_value(v, unit)
            p["amplitude"], p["amplitude_unit"] = a, u
        elif q == "frequency_hz":
            p["frequency_hz"] = v
        else:
            p["pulse_width_us"] = v
        p["train_duration_s"] = e.duration
        p["trial_type"] = "stimulation"
        for k, val in (mapping or {}).items():
            if k != "absolute":
                P.set_param(p, k, val)
    site = (rule.site if rule and rule.site else None) or (row.get("site") or "").strip() or None
    if site:
        p["site"] = site
    if rule and rule.contact:
        p["contact"] = rule.contact
    for k, val in (rule.set if rule else {}).items():
        if k != "absolute":
            P.set_param(p, k, val)
    return p, notes


def nwb_records(nf: I.NwbFile, source_format: str = "") -> tuple[list[M.Record], list[M.Record], list[str]]:
    """(stimulation records, marker records from an NWB `markers` table, notes)."""
    stim, marks, notes = [], [], []
    for name, cols in nf.intervals().items():
        desc = cols.pop("_description", "")
        n = len(cols.get("start_time", []))
        lname = name.lower()
        if lname == "markers":
            for i in range(n):
                marks.append({"onset": cols["start_time"][i],
                              "duration": (cols.get("stop_time", [None] * n)[i] or cols["start_time"][i])
                              - cols["start_time"][i],
                              "label": cols.get("label", [""] * n)[i], "trial_type": "marker"})
            continue
        param_cols = [c for c in cols if c not in ("start_time", "stop_time")
                      and P.classify_column(c).role not in ("extra", "recording", "onset", "stop")]
        if lname != "stimulation_source" and not ("stim" in lname or (lname in ("trials", "epochs") and param_cols)):
            continue
        if not param_cols and lname != "stimulation_source":
            continue
        src = "hdf5-attrs" if (source_format == "hdf5" or re.search(r"hdf5|attribute", desc, re.IGNORECASE)) \
            else "nwb"
        for i in range(n):
            row = {c: v[i] for c, v in cols.items() if c not in ("start_time", "stop_time")}
            prm, ext = P.from_record(row)
            st = float(cols["start_time"][i])
            sp = cols.get("stop_time")
            stop = float(sp[i]) if sp is not None and sp[i] is not None and math.isfinite(float(sp[i])) else None
            stim.append(M.Record(src, st, stop, prm, ext, f"NWB intervals/{name} row {i}"))
        notes.append(f"NWB intervals/{name}: {n} row(s), source {src}")
    return stim, marks, notes


def marker_records(rows: list[dict], pattern) -> tuple[list[M.Record], list[M.Record]]:
    """(stimulation records, baseline records) from imported markers."""
    if pattern == "none":
        return [], []
    cand = []
    for r in rows:
        label = str(r.get("label") or "").strip()
        if not label or label == "n/a":
            continue
        tt = str(r.get("trial_type") or "").strip().lower()
        if tt in ("stimulation",):  # already stimulation rows (e.g. a previous stim-events run): skip
            continue
        on = P.to_float(r.get("onset"))
        if on is None:
            continue
        dur = P.to_float(r.get("duration"))
        cand.append((on, dur, label))
    cand.sort()
    stim, base = [], []
    used_off = set()
    for i, (on, dur, label) in enumerate(cand):
        if i in used_off:
            continue
        if BASELINE_MARKER.search(label) and not STIM_MARKER.search(label):
            base.append(M.Record("marker", on, on + dur if dur else None, {"trial_type": "baseline", "label": label},
                                 origin=f"marker '{label}' at {on:g} s"))
            continue
        if pattern == "auto":
            prm = P.parse_tokens(label)
            if not (STIM_MARKER.search(label) or prm):
                continue
            ext = {}
        else:
            got = P.parse_named(pattern, label)
            if got is None:
                continue
            ext = got.pop("_extras", {})
            prm = got
        if OFF_WORD.search(label) and not ON_WORD.search(label):
            continue   # an "off" marker on its own closes nothing
        stop = on + dur if dur else None
        if ON_WORD.search(label) and stop is None:
            for j in range(i + 1, len(cand)):
                if OFF_WORD.search(cand[j][2]) and not ON_WORD.search(cand[j][2]):
                    stop = cand[j][0]
                    used_off.add(j)
                    break
                if ON_WORD.search(cand[j][2]):
                    break
        prm = dict(prm)
        prm["label"] = label
        stim.append(M.Record("marker", on, stop, prm, ext, f"marker '{label}' at {on:g} s"))
    return stim, base


def name_layer(rec: I.Recording, pattern) -> dict:
    if pattern == "none":
        return {}
    text = rec.source_path or rec.nwb.name
    if pattern == "auto":
        # tokens in the file and folder names, not in long free text
        out: dict = {}
        for part in Path(text.replace("\\", "/")).parts:
            for k, v in P.parse_tokens(Path(part).stem if "." in part else part).items():
                out.setdefault(k, v)
        return out
    got = P.parse_named(pattern, text)
    if got is None:
        got = P.parse_named(pattern, rec.recording_id)
    if got is None:
        return {}
    ext = got.pop("_extras", {})
    return {**got, **({"_extras": ext} if ext else {})}


def recording_end(nf: I.NwbFile, chans) -> float | None:
    end = None
    for ch in chans:
        try:
            if ch.has_timestamps:
                ts = nf.f[ch.h5path]["timestamps"]
                e = float(np.nanmax(ts[max(0, len(ts) - 1000):])) if len(ts) else None
            else:
                e = ch.starting_time + ch.n / float(ch.rate or 1.0)
        except Exception:  # noqa: BLE001
            e = None
        if e is not None and (end is None or e > end):
            end = e
    return end


def process(rec: I.Recording, prm: dict, smap: SM.StimMap, tables: TableSet, sds: dict, n_recs: int,
            markers: list[dict], qc: list) -> dict:
    """Events of one recording (list of row dicts) and what was found."""
    rid = rec.recording_id
    s: D.Settings = prm["settings"]
    t0 = time.monotonic()
    ch_events: list[M.Event] = []
    channel_info = []
    with I.NwbFile(rec.nwb) as nf:
        all_ch = nf.channels()
        rec_end = rec.duration_s or recording_end(nf, all_ch)
        for ch, row in I.stim_channels(nf, rec):
            original = (row.get("original_name") or "").strip()
            rule = smap.channel_rule(ch.name, original, ch.description)
            if rule and rule.kind == "ignore":
                qc.append([rid, "channel_kind", "ignore", f"{ch.name} ({original or ch.name}): stim-map.yaml"])
                continue
            unit = (rule.unit if rule and rule.unit else None) or (row.get("unit") or ch.unit or "").strip()
            st = D.scan(nf, ch)
            if rule and rule.kind:
                kind, why = rule.kind, "stim-map.yaml"
            else:
                kind, why = D.classify(ch.name, original, ch.description, unit, ch.rate, nf, ch, st)
            ss = D.Settings(**vars(s))
            if rule and rule.threshold is not None:
                if kind == "envelope":
                    ss.envelope_threshold = rule.threshold / (rule.scale or 1.0)
                else:
                    ss.trigger_threshold = rule.threshold / (rule.scale or 1.0)
            if kind == "envelope":
                eps, info = D.detect_envelope(nf, ch, st, ss)
            else:
                eps, info = D.detect_pulses(nf, ch, st, kind, ss)
            log.info("%s: channel %s (%s) %s [%s]: %d epoch(s)", rid, ch.name, original or "-", kind, why, len(eps))
            qc.append([rid, "channel_kind", kind, f"{ch.name} ({original or ch.name}), {ch.rate or 0:g} Hz, "
                                                  f"unit {unit or 'none'}: {why}; {len(eps)} epoch(s)"])
            channel_info.append({"channel": ch.name, "original_name": original, "kind": kind, "why": why,
                                 "unit": unit, "rate_hz": ch.rate, "n_epochs": len(eps),
                                 **{k: num(v) if isinstance(v, (int, float)) else v for k, v in info.items()}})
            if kind == "envelope" and any(e.sign < 0 for e in eps) and not (rule and rule.negative):
                qc.append([rid, "envelope_negative", ch.name,
                           "the envelope has negative values; they are written as negative amplitudes. Say what "
                           "negative means in stim-map.yaml (e.g. negative: {trial_type: block, waveform: kilohertz})"])
            for e in eps:
                p, _ = channel_layer(e, kind, unit, rule, row)
                ev = M.Event(onset=e.onset, duration=e.duration, group=(rule.group if rule else None),
                             timing=kind)
                ev.add(kind, p, origin=f"{kind} channel {ch.name}")
                ev.channel_epochs.append(e)
                ev.extras["_channel"] = original or ch.name
                for n in e.notes:
                    qc.append([rid, "channel_note", ch.name, f"train at {e.onset:.3f} s: {n}"])
                ch_events.append(ev)
        nwb_stim, nwb_marks, nwb_notes = nwb_records(nf, rec.source_format)
    # trigger + monitor (or channels of one stim-map group) describing the same trains: one event
    ch_events.sort(key=lambda e: e.onset)
    merged: list[M.Event] = []
    for ev in ch_events:
        kind = ev.timing
        hit = None
        for m in merged[-8:]:
            same_group = ev.group is not None and ev.group == m.group
            pair = {kind, m.timing} == {"trigger", "monitor"} or ("monitor" in m.sources() and kind == "trigger")
            if not (same_group or pair) or ev.extras["_channel"] == m.extras.get("_channel"):
                continue
            ov = M._overlap(ev.onset, ev.end, m.onset, m.end)
            shorter = max(min(ev.duration or 0, m.duration or 0), 1e-3)
            if ov >= 0.5 * shorter or abs(ev.onset - m.onset) <= 2 * s.pulse_merge_s:
                hit = m
                break
        if hit is None:
            merged.append(ev)
            continue
        if kind == "trigger" and hit.timing != "trigger":
            hit.onset, hit.duration, hit.timing = ev.onset, ev.duration, "trigger"
        for src, p in ev.layers:
            hit.add(src, p)
        hit.channel_epochs.extend(ev.channel_epochs)
        hit.extras["_channel"] = hit.extras["_channel"] + "+" + ev.extras["_channel"]
    events = merged
    has_channel = bool(events)

    # timed records: NWB per-trial table, parameter tables, markers
    for n in nwb_notes:
        log.info("%s: %s", rid, n)
    marker_rows = markers or nwb_marks
    mk_stim, mk_base = marker_records(marker_rows, prm["marker_pattern"])
    table_sets, _ = tables.records_for(rec, n_recs)
    record_sources = []
    unmatched_all: list[M.Record] = []

    def attach(records: list[M.Record], label: str) -> None:
        nonlocal events
        stim_recs = [r for r in records if r.params.get("trial_type") != "baseline"]
        base_recs = [r for r in records if r.params.get("trial_type") == "baseline"]
        pairs, left = M.match_records(stim_recs, events, prm["tol"])
        for r, ev in pairs:
            ev.add(r.source, r.params, r.extras, r.origin)
        for r in left:
            if r.start is None:
                unmatched_all.append(r)
                continue
            ev = M.event_from_record(r)
            events.append(ev)
            if has_channel:
                qc.append([rid, "unmatched_row", r.source,
                           f"{r.origin}: no detected train within {prm['tol']:g} s (or inside its window); placed at "
                           "its own time"])
        for r in base_recs:
            if r.start is not None:
                events.append(M.event_from_record(r))
        if records:
            record_sources.append(label)
        events.sort(key=lambda e: e.onset)

    attach(nwb_stim, "nwb")
    for mode, recs in table_sets:
        if mode in ("time",) or (mode == "auto" and any(r.start is not None for r in recs)):
            attach(recs, "table")
        elif mode == "recording" or (mode == "auto" and len(recs) == 1 and len(events) != 1):
            for ev in events:
                ev.add("table", recs[0].params, recs[0].extras, recs[0].origin)
            if not events:
                unmatched_all.append(recs[0])
            record_sources.append("table")
        else:  # order
            targets = [e for e in events if e.resolved().get("trial_type", "stimulation") in ("stimulation", "block")]
            if len(recs) != len(targets):
                qc.append([rid, "row_count", f"{len(recs)} vs {len(targets)}",
                           f"parameter table rows ({len(recs)}) and detected trains ({len(targets)}) differ in number; "
                           "matched in order as far as both go"])
            for r, ev in zip(recs, targets):
                ev.add("table", r.params, r.extras, r.origin)
            for r in recs[len(targets):]:
                unmatched_all.append(r)
            record_sources.append("table")
    attach(mk_stim + mk_base, "marker")

    # recording-level layers: file/folder names, SDS, stim-map defaults
    rec_layers = []
    nl = name_layer(rec, prm["name_pattern"])
    if nl:
        ext = nl.pop("_extras", {})
        rec_layers.append(("filename", nl, ext))
    if prm["use_sds"] and sds:
        sp, used = sds_layer(rec, sds)
        if sp:
            rec_layers.append(("sds", sp, {}))
    if smap.defaults:
        rec_layers.append(("user", dict(smap.defaults), {}))
    for r in unmatched_all:   # untimed rows that matched no train: recording-level too
        rec_layers.insert(0, (r.source, r.params, r.extras))
    stimish = [e for e in events if e.resolved().get("trial_type", "stimulation") != "baseline"]
    if not stimish:
        useful = [(src, p, ext) for src, p, ext in rec_layers
                  if any(k in p for k in ("amplitude", "frequency_hz", "pulse_width_us", "train_duration_s",
                                          "condition", "waveform", "polarity"))]
        if useful:
            ev = M.Event(onset=0.0, duration=rec_end, timing="recording")
            for src, p, ext in useful:
                ev.add(src, {**p, "trial_type": "condition"}, ext)
            events.append(ev)
            qc.append([rid, "no_timing", "condition",
                       "no channel, table time or marker places the stimulation: one `condition` row spans the "
                       "recording with the parameters from " + ", ".join(sorted({u[0] for u in useful}))])
    else:
        for ev in events:
            for src, p, ext in rec_layers:
                ev.add(src, p, ext)

    # rows
    events.sort(key=lambda e: e.onset)
    counters: dict[str, int] = {}
    prefix = {"stimulation": "stim", "block": "block", "baseline": "base", "condition": "cond"}
    rows = []
    for ev in events:
        r = ev.resolved()
        tt = r.get("trial_type") or "stimulation"
        pf = prefix.get(tt, "ev")
        counters[pf] = counters.get(pf, 0) + 1
        dur = ev.duration
        row = {"recording_id": rid, "onset": round(ev.onset, 6), "duration": round(dur, 6) if dur is not None else None,
               "trial_type": tt, "event_id": f"{pf}-{counters[pf]:04d}", "source": "|".join(ev.sources())}
        for k in P.PARAMS:
            if k != "trial_type":
                row[k] = r.get(k)
        if row["train_duration_s"] is None and tt in ("stimulation", "block") and dur is not None:
            row["train_duration_s"] = round(dur, 6)
        if row.get("amplitude") is not None and not row.get("amplitude_unit"):
            row["amplitude_unit"] = "n/a"
        for k, v in ev.extras.items():
            if not k.startswith("_"):
                row[k] = v
        row["_channel"] = ev.extras.get("_channel", "")
        rows.append(row)
        for name, hv, cv, hs, cs in M.disagreements(ev):
            qc.append([rid, "table_vs_channel", name,
                       f"{row['event_id']} at {ev.onset:.3f} s: {hs} says {hv:.6g}, the {cs} channel {cv:.6g} "
                       "(the table value is written)"])
    # overlaps and baseline gaps
    act = [r for r in rows if r["trial_type"] in ("stimulation", "block")]
    overlaps = set()
    prev_end = None
    prev_id = None
    for r in act:
        on = r["onset"]
        end = on + (r["duration"] or 0.0)
        if prev_end is not None and on < prev_end - 1e-6:
            overlaps.update({r["event_id"], prev_id})
            qc.append([rid, "overlap", round(prev_end - on, 3),
                       f"{r['event_id']} starts {prev_end - on:.3f} s before {prev_id} ends"])
        gap = on - (prev_end if prev_end is not None else 0.0)
        if gap < prm["min_gap_s"] and not (prev_end is not None and on < prev_end - 1e-6):
            what = f"after {prev_id} ends" if prev_end is not None else "after the recording starts"
            qc.append([rid, "baseline_gap", round(gap, 3),
                       f"{r['event_id']} starts {gap:.2f} s {what} (MIN_GAP_S {prm['min_gap_s']:g} s): a baseline "
                       "window before it would include "
                       + ("the previous stimulation or its after-effects" if prev_end is not None
                          else "too little recording")])
        if prev_end is None or end > prev_end:
            prev_end, prev_id = end, r["event_id"]
    for r in rows:
        if r.get("amplitude") is not None and r.get("amplitude_unit") == "n/a":
            qc.append([rid, "amplitude_unit_unknown", r["event_id"],
                       "the amplitude's unit is not known (no physical unit on the channel or in the table); give it "
                       "in stim-map.yaml (unit:) to compare amplitudes across recordings"])
            break
    sources = sorted({s_ for r in rows for s_ in r["source"].split("|")}, key=lambda x: M.RANK.get(x, 9))
    status = "ok"
    reason = None
    if not rows:
        status = "skipped"
        reason = ("no stimulation found: no stimulation channel with pulses or epochs, no parameter table rows, no "
                  "stimulation markers and no parameters in the file or folder names")
    return {"rows": rows, "status": status, "reason": reason, "seconds": round(time.monotonic() - t0, 2),
            "channels": channel_info, "sources": sources, "rec_end": rec_end, "overlaps": overlaps,
            "records": record_sources}


# --------------------------------------------------------------------------- summaries

def varied(rows: list[dict]) -> list[str]:
    st = [r for r in rows if r["trial_type"] == "stimulation"]
    out = []
    for k in VARY_KEYS:
        vals = {(round(r[k], 6) if isinstance(r[k], float) else r[k]) for r in st if r.get(k) is not None}
        if len(vals) > 1:
            out.append(k)
    return out


def rng(rows, k):
    vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return (min(vals), max(vals)) if vals else (None, None)


def peak_rss_mb() -> float:
    for f in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return round(int(Path(f).read_text().strip()) / 2**20, 1)
        except (OSError, ValueError):
            continue
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2**20 if sys.platform == "darwin" else r / 1024, 1)


def main() -> int:
    t_start = time.monotonic()
    started = datetime.now(timezone.utc)
    input_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    try:
        raw, prm = read_params()
    except ValueError as e:
        log.error("parameters: %s", e)
        return 2
    out = output_dir / APP_NAME
    out.mkdir(parents=True, exist_ok=True)
    log.info("%s %s: INPUT_DIR=%s OUTPUT_DIR=%s threads=%d (%s)", APP_NAME, app_version(), input_dir, out,
             _THREADS, _THREADS_WHY)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s does not exist", input_dir)
        return 1
    physio = I.find_physio(input_dir)
    physio_files = {r.nwb.resolve() for r in physio.recordings}
    params_in = I.find_params(input_dir, physio_files)
    try:
        smap = SM.load(params_in.stim_map)
    except ValueError as e:
        log.error("stim-map.yaml: %s", e)
        return 2
    if params_in.stim_map:
        log.info("stim-map: %s (%d channel rule(s), %d table rule(s))", params_in.stim_map.name, len(smap.channels),
                 len(smap.tables))
    tables = TableSet(params_in.tables, input_dir, smap)
    if tables.tables:
        log.info("parameter tables: %s", ", ".join(t["rel"] for t in tables.tables))
    if params_in.sds:
        log.info("SDS metadata: %s%s", ", ".join(p.name for p in params_in.sds.values()),
                 "" if prm["use_sds"] else " (not used: USE_SDS=false)")
    if not physio.recordings:
        log.error("no recording found under %s (recordings.csv or *.nwb)", input_dir)
        return 1
    log.info("%d recording(s) found", len(physio.recordings))

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(input_dir))
        except ValueError:
            return str(p)

    all_rows: list[dict] = []
    qc: list = []
    summaries, features, figures, warnings_out = [], [], [], []
    per_rec, headlines = {}, {}
    n_ok = n_failed = 0
    figdir = out / "figures"
    for i, rec in enumerate(physio.recordings, 1):
        log.info("[%d/%d] %s (%s)", i, len(physio.recordings), rec.recording_id, rel(rec.nwb))
        try:
            res = process(rec, prm, smap, tables, params_in.sds if prm["use_sds"] else {}, len(physio.recordings),
                          physio.markers.get(rec.recording_id, []), qc)
        except Exception as e:  # noqa: BLE001
            log.error("%s: failed: %s", rec.recording_id, e)
            log.info(traceback.format_exc())
            per_rec[rec.recording_id] = {"status": "failed", "error": str(e), "path": rel(rec.nwb)}
            warnings_out.append(f"{rec.recording_id}: failed: {e}")
            n_failed += 1
            continue
        rows = res["rows"]
        rid = rec.recording_id
        vary = varied(rows)
        st_rows = [r for r in rows if r["trial_type"] == "stimulation"]
        n = {t: sum(1 for r in rows if r["trial_type"] == t) for t in ("stimulation", "block", "baseline", "condition")}
        amin, amax = rng(st_rows, "amplitude")
        units = sorted({r["amplitude_unit"] for r in st_rows if r.get("amplitude_unit")})
        fmin, fmax = rng(st_rows, "frequency_hz")
        pmin, pmax = rng(st_rows, "pulse_width_us")
        dmin, dmax = rng(st_rows, "train_duration_s")
        total = sum((r["duration"] or 0.0) for r in st_rows)
        summaries.append([rid, res["status"], n["stimulation"], n["block"], n["baseline"], n["condition"],
                          "|".join(vary), amin, amax, "|".join(units), fmin, fmax, pmin, pmax, dmin, dmax,
                          round(total, 3), "|".join(res["sources"])])
        for feat, val in (("stim_n_trains", n["stimulation"]), ("stim_n_blocks", n["block"]),
                          ("stim_total_s", round(total, 3)), ("stim_amplitude_min", amin), ("stim_amplitude_max", amax),
                          ("stim_frequency_hz_min", fmin), ("stim_frequency_hz_max", fmax),
                          ("stim_n_amplitude_levels", len({r["amplitude"] for r in st_rows if r.get("amplitude") is not None})),
                          ("stim_n_frequency_levels", len({round(r["frequency_hz"], 3) for r in st_rows
                                                           if r.get("frequency_hz") is not None}))):
            if val is not None:
                features.append([rid, "", feat, val])
        per_rec[rid] = {"status": res["status"], "seconds": res["seconds"], "path": rel(rec.nwb),
                        "species": rec.species or None, "channels": res["channels"], "sources": res["sources"]}
        if res["reason"]:
            per_rec[rid]["reason"] = res["reason"]
            warnings_out.append(f"{rid}: skipped: {res['reason']}")
        else:
            n_ok += 1
        headlines[rid] = {"n_stimulations": n["stimulation"], "n_blocks": n["block"], "n_baseline": n["baseline"],
                          "parameters_varied": vary, "amplitude_min": num(amin), "amplitude_max": num(amax),
                          "amplitude_unit": "|".join(units) or None, "frequency_hz_min": num(fmin),
                          "frequency_hz_max": num(fmax), "total_stimulation_s": num(total, 3),
                          "sources": res["sources"], "n_qc_flags": sum(1 for q in qc if q[0] == rid and
                                                                       q[1] not in ("channel_kind",))}
        log.info("%s: %d stimulation(s), %d block(s), %d baseline, %d condition; varied: %s; sources: %s (%.1fs)",
                 rid, n["stimulation"], n["block"], n["baseline"], n["condition"], ", ".join(vary) or "-",
                 ", ".join(res["sources"]) or "-", res["seconds"])
        all_rows.extend(rows)
        if prm["figures"] and rows:
            try:
                from processor import figures as F  # noqa: PLC0415
                figdir.mkdir(exist_ok=True)
                name = f"{rid}_timeline.png"
                F.timeline(figdir / name, rid, [{k: P.fmt(v) if k != "_channel" else v for k, v in r.items()}
                                                for r in rows], vary, res["rec_end"], res["overlaps"])
                figures.append({"path": f"figures/{name}", "caption": (
                    f"{rid}: every stimulation train found, as a bar from its onset to its end on the recording's time "
                    "axis" + (", one row per pulse frequency" if "frequency_hz" in vary else ", one row per source")
                    + ", coloured by amplitude" + (" (hatched grey: block epochs, e.g. kilohertz block)" if n["block"] else "")
                    + ". Check that the number and order of trains match the protocol, that amplitudes step as "
                    "planned, and look for red outlines: trains that overlap another one. "
                    + (f"Parameters that vary: {', '.join(vary)}. " if vary else "")
                    + F.RESEARCH)})
            except Exception as e:  # noqa: BLE001
                log.warning("%s: figure failed: %s", rid, e)

    # tables
    extra_cols = sorted({k for r in all_rows for k in r if k not in P.EVENT_COLUMNS and not k.startswith("_")})
    with open(out / "events.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(P.EVENT_COLUMNS + extra_cols)
        for r in all_rows:
            w.writerow([P.fmt(r.get(c)) for c in P.EVENT_COLUMNS + extra_cols])
    for name, header, data in (("stim_summary.csv", SUMMARY_HEADER, summaries), ("qc.csv", QC_HEADER, qc),
                               ("features.csv", FEATURE_HEADER, features)):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(header)
            for row in data:
                w.writerow(["" if v is None else (P.fmt(v) if isinstance(v, float) else v) for v in row])

    finished = datetime.now(timezone.utc)
    inputs = physio.sources + [rel(p) for p in params_in.tables] + \
        ([rel(params_in.stim_map)] if params_in.stim_map else []) + [rel(p) for p in params_in.sds.values()]
    provenance = {
        "app": APP_NAME, "version": app_version(), "variant": None, "tools": tool_versions(),
        "parameters": raw, "inputs": sorted(set(inputs)),
        "started": started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "finished": finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t_start, 2), "peak_rss_mb": peak_rss_mb(),
        "threads": {"requested": int(raw["THREADS"]), "used": _THREADS, "why": _THREADS_WHY},
        "recordings": per_rec, "citations": CITATIONS, "licence_notice": None,
    }
    with open(out / "provenance.json", "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, ensure_ascii=False, default=str)
    if not all_rows:
        warnings_out.append("no stimulation events found in any recording: events.tsv has the header only")
    summary = {
        "app": APP_NAME, "version": app_version(), "recordings": headlines, "figures": figures,
        "tables": [
            {"path": "events.tsv", "caption": "One row per stimulation train (and block, baseline or condition "
                                              "epoch) with its parameters and where each came from (source)."},
            {"path": "stim_summary.csv", "caption": "Per recording: number of trains, which parameters vary and "
                                                    "their ranges."},
            {"path": "qc.csv", "caption": "Checks: how each stimulation channel was read, overlapping trains, trains "
                                          f"less than {prm['min_gap_s']:g} s after the previous one (no clean baseline), "
                                          "table vs channel disagreements, table rows without a detected train."},
            {"path": "features.csv", "caption": "Per-recording stimulation counts and ranges in the shared long shape."},
        ],
        "warnings": warnings_out,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    log.info("%d recording(s) with events, %d skipped, %d failed; %d event row(s); %.1fs, peak memory %.0f MB; wrote %s",
             n_ok, len(physio.recordings) - n_ok - n_failed, n_failed, len(all_rows), time.monotonic() - t_start,
             provenance["peak_rss_mb"], out)
    if n_failed and n_ok == 0:
        log.error("every recording with input failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
