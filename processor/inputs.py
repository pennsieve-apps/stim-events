"""
Finding and reading the inputs.

physio port (physio-import's output, or any NWB files): recordings.csv, channels.csv, events.tsv and
one NWB file per recording, found by name and content, recursively, following symlinks.

params port (optional): parameter tables (CSV/TSV/XLSX), a stim-map.yaml, and SDS metadata
(manifest.xlsx, samples.xlsx, performances.xlsx) when present.

NWB files are read with h5py directly (NWB 2.x is HDF5 with a fixed layout): channel data by slice,
never a whole high-rate channel at once, and the `intervals` tables as columns. pynwb is not needed at
run time, which keeps the image small; the tests check the reader against files written by pynwb.
"""
from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger("stim-events")

NWB_SUFFIXES = (".nwb",)
TABLE_SUFFIXES = (".csv", ".tsv", ".xlsx", ".xlsm")
SDS_NAMES = ("manifest", "samples", "performances", "subjects")
# stimulation-line names when there is no channels.csv (conventions.md: a bare "pulse" is a finger
# pulse / PPG transducer, not a stimulus line)
STIM_NAME = re.compile(r"stim|trig|ttl|marker|pulse.?train|sync|timing|ctrl.?amp", re.IGNORECASE)


# --------------------------------------------------------------------------- file discovery

def walk(root: Path):
    """Every file under root, recursively, following symlinks (INPUT_DIR entries are symlinks); no loops."""
    seen: set = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        try:
            st = os.stat(dirpath)
            key = (st.st_dev, st.st_ino)
        except OSError:
            continue
        if key in seen:
            dirnames[:] = []
            continue
        seen.add(key)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(filenames):
            if not f.startswith((".", "~$")):
                yield Path(dirpath) / f


def read_delimited(path: Path) -> list[dict]:
    """CSV or TSV (by extension, else sniffed) as a list of row dicts; UTF-8 with or without BOM."""
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()
    if not text.strip():
        return []
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    if path.suffix.lower() not in (".tsv",):
        first = text.splitlines()[0]
        if first.count("\t") > first.count(",") and first.count("\t") > first.count(";"):
            delim = "\t"
        elif first.count(";") > first.count(","):
            delim = ";"
    rows = list(csv.DictReader(text.splitlines(), delimiter=delim))
    return [{(k or "").strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k} for r in rows]


def read_xlsx(path: Path, sheet=None) -> list[dict]:
    """The first sheet (or `sheet`, a name or index) of an XLSX as row dicts; the header is the first
    non-empty row."""
    import openpyxl  # noqa: PLC0415 - only when an XLSX is given

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet is None:
            ws = wb.worksheets[0]
        elif isinstance(sheet, int):
            ws = wb.worksheets[sheet]
        else:
            ws = wb[str(sheet)]
        header = None
        out = []
        for row in ws.iter_rows(values_only=True):
            if header is None:
                if row and any(c not in (None, "") for c in row):
                    header = [str(c).strip() if c is not None else "" for c in row]
                continue
            if not row or all(c in (None, "") for c in row):
                continue
            out.append({h: v for h, v in zip(header, row) if h})
        return out
    finally:
        wb.close()


def read_table(path: Path, sheet=None) -> list[dict]:
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return read_xlsx(path, sheet)
    return read_delimited(path)


# --------------------------------------------------------------------------- NWB (h5py)

def _s(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray) and v.shape == ():
        return _s(v[()])
    return str(v)


@dataclass
class Channel:
    name: str                  # NWB TimeSeries name (column k of a 2-D series: name[k])
    h5path: str                # path of the TimeSeries group
    column: int | None         # column of 2-D data, else None
    n: int
    rate: float | None         # Hz, None when only timestamps
    starting_time: float
    has_timestamps: bool
    unit: str
    conversion: float
    offset: float
    description: str
    in_stimulus: bool          # under /stimulus/presentation

    def time_of(self, idx: np.ndarray, f: h5py.File) -> np.ndarray:
        if self.has_timestamps:
            return np.asarray(f[self.h5path]["timestamps"][int(idx.min()):int(idx.max()) + 1])[idx - idx.min()]
        return self.starting_time + idx / float(self.rate)


class NwbFile:
    """A thin, read-only view of one NWB file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.f = h5py.File(self.path, "r")

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    @property
    def identifier(self) -> str:
        return _s(self.f["identifier"][()]) if "identifier" in self.f else self.path.stem

    @property
    def session_start(self) -> str:
        return _s(self.f["session_start_time"][()]) if "session_start_time" in self.f else ""

    @property
    def species(self) -> str:
        try:
            return _s(self.f["general/subject/species"][()])
        except KeyError:
            return ""

    @property
    def subject_id(self) -> str:
        try:
            return _s(self.f["general/subject/subject_id"][()])
        except KeyError:
            return ""

    def channels(self) -> list[Channel]:
        out = []
        for base, in_stim in (("acquisition", False), ("stimulus/presentation", True)):
            if base not in self.f:
                continue
            for name, obj in self.f[base].items():
                if not isinstance(obj, h5py.Group) or "data" not in obj:
                    continue
                data = obj["data"]
                if not isinstance(data, h5py.Dataset) or data.dtype.kind not in "iufb" or data.ndim not in (1, 2):
                    continue
                rate, start, has_ts = None, 0.0, "timestamps" in obj
                if "starting_time" in obj:
                    st = obj["starting_time"]
                    start = float(st[()])
                    rate = float(st.attrs.get("rate", 0)) or None
                if has_ts and rate is None:
                    ts = obj["timestamps"]
                    k = min(len(ts), 2001)
                    if k >= 2:
                        d = np.diff(np.asarray(ts[:k], dtype=float))
                        d = d[d > 0]
                        rate = float(1.0 / np.median(d)) if len(d) else None
                    start = float(ts[0]) if len(ts) else 0.0
                common = dict(h5path=obj.name, n=int(data.shape[0]), rate=rate, starting_time=start,
                              has_timestamps=has_ts, unit=_s(data.attrs.get("unit", "")),
                              conversion=float(data.attrs.get("conversion", 1.0)),
                              offset=float(data.attrs.get("offset", 0.0)),
                              description=_s(obj.attrs.get("description", "")), in_stimulus=in_stim)
                if data.ndim == 2 and data.shape[1] > 1:
                    for k in range(data.shape[1]):
                        out.append(Channel(name=f"{name}[{k}]", column=k, **common))
                else:
                    out.append(Channel(name=name, column=0 if data.ndim == 2 else None, **common))
        return out

    def read(self, ch: Channel, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray]:
        """(times s, values in the channel's unit) of samples i0..i1-1; NaN kept."""
        g = self.f[ch.h5path]
        d = g["data"]
        x = d[i0:i1] if ch.column is None else d[i0:i1, ch.column]
        x = np.asarray(x, dtype=np.float64)
        if ch.conversion != 1.0:
            x *= ch.conversion
        if ch.offset:
            x += ch.offset
        if ch.has_timestamps:
            t = np.asarray(g["timestamps"][i0:i1], dtype=np.float64)
        else:
            t = ch.starting_time + np.arange(i0, i1, dtype=np.float64) / float(ch.rate or 1.0)
        return t, x

    def intervals(self) -> dict[str, dict[str, list]]:
        """Every `intervals` table: {name: {column: values}} (ragged columns joined with '|')."""
        out: dict = {}
        if "intervals" not in self.f:
            return out
        for name, g in self.f["intervals"].items():
            if not isinstance(g, h5py.Group) or "start_time" not in g:
                continue
            cols: dict = {}
            n = len(g["start_time"])
            for c, ds in g.items():
                if not isinstance(ds, h5py.Dataset) or c == "id" or c.endswith("_index"):
                    continue
                if c + "_index" in g:  # ragged column
                    idx = np.asarray(g[c + "_index"][:])
                    flat = ds[:]
                    vals, lo = [], 0
                    for hi in idx:
                        part = flat[lo:int(hi)]
                        vals.append("|".join(_s(v) for v in part))
                        lo = int(hi)
                    cols[c] = vals
                    continue
                if ds.ndim != 1 or len(ds) != n:
                    continue
                if ds.dtype.kind in "OSU":
                    cols[c] = [_s(v) for v in ds[:]]
                elif ds.dtype.kind in "iufb":
                    cols[c] = [v.item() if hasattr(v, "item") else v for v in ds[:]]
                elif ds.dtype.kind == "V":  # references (e.g. timeseries column): not parameters
                    continue
            cols["_description"] = _s(g.attrs.get("description", ""))
            out[name] = cols
        return out


# --------------------------------------------------------------------------- physio dataset

@dataclass
class Recording:
    recording_id: str
    nwb: Path
    subject: str = ""
    session: str = ""
    species: str = ""
    source_path: str = ""
    source_format: str = ""
    duration_s: float | None = None
    channels: dict[str, dict] = field(default_factory=dict)   # channels.csv rows by NWB channel name


@dataclass
class PhysioInput:
    recordings: list[Recording]
    markers: dict[str, list[dict]]        # events.tsv rows by recording_id (physio-import's markers)
    sources: list[str]                    # files read (relative paths for provenance)


def find_physio(root: Path) -> PhysioInput:
    """recordings.csv (+ channels.csv, events.tsv) when present, else every NWB file found."""
    files = list(walk(root))
    nwbs = [p for p in files if p.suffix.lower() in NWB_SUFFIXES]
    rec_csvs = [p for p in files if p.name == "recordings.csv"]
    ch_csvs = [p for p in files if p.name == "channels.csv"]
    # physio-import's markers sit next to recordings.csv; our own events.tsv (stim-events/) is not an input
    ev_tsvs = [p for p in files if p.name == "events.tsv" and "stim-events" not in p.parts]
    used: list[Path] = []
    recs: dict[str, Recording] = {}
    by_nwb: dict[Path, Recording] = {}
    for rc in rec_csvs:
        used.append(rc)
        for r in read_delimited(rc):
            rid = (r.get("recording_id") or "").strip()
            rel = (r.get("path") or "").strip()
            if not rid or not rel:
                continue
            p = (rc.parent / rel)
            if not p.exists():
                cands = [q for q in nwbs if q.name == Path(rel).name]
                if not cands:
                    log.warning("recordings.csv lists %s but %s was not found", rid, rel)
                    continue
                p = cands[0]
            dur = None
            try:
                dur = float(r.get("duration_s") or "nan")
                dur = dur if np.isfinite(dur) else None
            except ValueError:
                pass
            rec = Recording(rid, p, subject=r.get("subject", "") or "", session=r.get("session", "") or "",
                            species=r.get("species", "") or "", source_path=r.get("source_path", "") or "",
                            source_format=(r.get("source_format") or "").strip().lower(), duration_s=dur)
            recs[rid] = rec
            by_nwb[p.resolve()] = rec
    for p in nwbs:
        if p.resolve() in by_nwb:
            continue
        try:
            with NwbFile(p) as nf:
                rid, subj, sp = nf.identifier, nf.subject_id, nf.species
        except Exception as e:  # noqa: BLE001
            log.warning("%s: not a readable NWB file (%s)", p, e)
            continue
        rid = re.sub(r"[^A-Za-z0-9_-]+", "-", rid) or p.stem
        if rid in recs:
            rid = f"{rid}-{p.stem}"
        recs[rid] = Recording(rid, p, subject=subj, species=sp, source_path=str(p.relative_to(root)) if
                              str(p).startswith(str(root)) else p.name)
        by_nwb[p.resolve()] = recs[rid]
    for cc in ch_csvs:
        used.append(cc)
        for r in read_delimited(cc):
            rec = recs.get((r.get("recording_id") or "").strip())
            if rec is not None and r.get("channel"):
                rec.channels[r["channel"].strip()] = r
    markers: dict[str, list[dict]] = {}
    for et in ev_tsvs:
        used.append(et)
        for r in read_delimited(et):
            rid = (r.get("recording_id") or "").strip()
            if rid in recs:
                markers.setdefault(rid, []).append(r)
    rels = []
    for p in used + [r.nwb for r in recs.values()]:
        try:
            rels.append(str(p.relative_to(root)))
        except ValueError:
            rels.append(str(p))
    return PhysioInput(sorted(recs.values(), key=lambda r: r.recording_id), markers, sorted(set(rels)))


def stim_channels(nf: NwbFile, rec: Recording) -> list[tuple[Channel, dict]]:
    """The stimulation channels of a recording: channels.csv type `stim`; without channels.csv, names
    `stim*` (physio-import's naming), names that look like a stimulus line, or anything under
    /stimulus/presentation. A processed copy of a stimulation channel (channels.csv `derived_from`
    naming another stim channel of the recording) is left out: the raw channel is read instead, so a
    train is never counted twice; a derived channel without a raw sibling (`derived_from` unknown or
    naming a channel that is not there) is used."""
    out = []
    for ch in nf.channels():
        base = ch.name.split("[")[0]
        row = rec.channels.get(base) or {}
        typ = (row.get("type") or "").strip().lower()
        if rec.channels:
            is_stim = typ == "stim"
        else:
            is_stim = (ch.in_stimulus or base.lower() == "stim" or base.lower().startswith("stim-")
                       or bool(STIM_NAME.search(base)) or bool(STIM_NAME.search(ch.description)))
        if is_stim:
            out.append((ch, row))
    names = {ch.name.split("[")[0] for ch, _ in out}
    kept = []
    for ch, row in out:
        src = (row.get("derived_from") or "").strip()
        if src and src.lower() != "unknown" and src in names and src != ch.name.split("[")[0]:
            log.info("%s: channel %s is derived from %s: the raw channel is used", rec.recording_id, ch.name, src)
            continue
        kept.append((ch, row))
    return kept


# --------------------------------------------------------------------------- params port

@dataclass
class ParamsInput:
    tables: list[Path]
    stim_map: Path | None
    sds: dict[str, Path]          # 'manifest' / 'samples' / 'performances' / 'subjects' -> path


def find_params(root: Path, physio_files: set[Path]) -> ParamsInput:
    """Parameter tables, stim-map.yaml and SDS metadata under root, excluding the physio dataset's own
    index tables (recordings.csv, channels.csv, events.tsv, qc.csv, ...)."""
    tables, sds, stim_map = [], {}, None
    skip_names = {"recordings.csv", "channels.csv", "events.tsv", "qc.csv", "channels-map.csv",
                  "recordings-map.csv", "stim_summary.csv", "features.csv", "series.csv", "beats.csv", "hrv.csv",
                  "bp_beats.csv", "breaths.csv", "brs.csv", "rsa.csv", "responses.csv", "fits.csv",
                  "surface.csv", "fulcrum.csv", "dataset_description.xlsx", "code_description.xlsx",
                  "submission.xlsx", "resources.xlsx"}
    for p in walk(root):
        if p.resolve() in physio_files:
            continue
        name = p.name.lower()
        if name in ("stim-map.yaml", "stim-map.yml", "stim_map.yaml", "stim_map.yml"):
            if stim_map is None:
                stim_map = p
            else:
                log.warning("more than one stim-map.yaml: using %s, ignoring %s", stim_map, p)
            continue
        if p.suffix.lower() not in TABLE_SUFFIXES or name in skip_names:
            continue
        if any(part in ("physio-import", "stim-events", "ecg-hrv", "bp-resp", "stim-dose-response")
               for part in p.parts[len(root.parts):]):
            continue
        stem = p.stem.lower()
        if stem in SDS_NAMES and p.suffix.lower() in (".xlsx", ".csv", ".tsv"):
            sds.setdefault(stem, p)
            continue
        tables.append(p)
    return ParamsInput(sorted(tables), stim_map, sds)
