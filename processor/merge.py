"""
Merging the sources of one recording into events, and the QC checks.

Priority (highest first; conventions.md and the builder brief): the NWB `stimulation_source` table
(per-trial parameters copied from the source file), a parameter table in the params port, the
stimulation channels (trigger, monitor, envelope), imported markers, file/folder names, SDS
metadata, stim-map.yaml defaults. A higher source's value wins; a lower one only fills what is
missing. Timing comes from a channel whenever a table row, NWB row or marker was matched to a
detected train (a channel places a train to the sample; tables usually give a trial window or a
rounded time); the row's own times are used when no channel train matches it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from processor import params as P

RANK = {"nwb": 0, "hdf5-attrs": 0, "table": 1, "trigger": 2, "monitor": 2, "envelope": 2, "marker": 3,
        "filename": 4, "sds": 5, "user": 6}
CHANNEL_SOURCES = ("trigger", "monitor", "envelope")


@dataclass
class Record:
    """A row of a timed or ordered source (NWB interval table, parameter table, marker)."""
    source: str                    # nwb | hdf5-attrs | table | marker
    start: float | None            # seconds from the recording start (None: no time, order-matched)
    stop: float | None
    params: dict
    extras: dict = field(default_factory=dict)
    origin: str = ""               # where it came from, for QC notes (file, row)
    expected: float | None = None  # expected train onset (window start + the windows' common offset)


@dataclass
class Event:
    onset: float
    duration: float | None
    layers: list[tuple[str, dict]] = field(default_factory=list)   # (source, params), any order
    extras: dict = field(default_factory=dict)
    channel_epochs: list = field(default_factory=list)
    origins: list[str] = field(default_factory=list)
    group: str | None = None
    timing: str = ""               # which source gave onset/duration

    def add(self, source: str, params: dict, extras: dict | None = None, origin: str = "") -> None:
        self.layers.append((source, dict(params)))
        for k, v in (extras or {}).items():
            self.extras.setdefault(k, v)
        if origin:
            self.origins.append(origin)

    def sources(self) -> list[str]:
        seen = []
        for s, _ in sorted(self.layers, key=lambda sp: RANK.get(sp[0], 9)):
            if s not in seen:
                seen.append(s)
        return seen

    def resolved(self) -> dict:
        """Parameters by priority: the highest-ranked source that has a value wins."""
        out: dict = {}
        for src, prm in sorted(self.layers, key=lambda sp: RANK.get(sp[0], 9)):
            for k, v in prm.items():
                if v is None or k in out:
                    continue
                if k == "amplitude_unit" and "amplitude" in out and out.get("amplitude_unit"):
                    continue
                if k == "amplitude":
                    out["amplitude"] = v
                    if prm.get("amplitude_unit"):
                        out["amplitude_unit"] = prm["amplitude_unit"]
                    continue
                out[k] = v
        return out

    @property
    def end(self) -> float:
        return self.onset + (self.duration or 0.0)


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def window_offset(records: list[Record], events: list[Event], tol: float) -> tuple[float | None, int, int]:
    """
    For records that give a trial window (start, stop) rather than the train's own time: the offset
    of the train within its window that most rows agree on (trains start at window start + offset),
    the number of rows that have a train at that offset, and the number of window rows. The offset
    is the densest tol-wide cluster of (train onset - window start) over all trains in all windows,
    counted by distinct rows; None when fewer than half the rows (or fewer than 2) agree.
    """
    win = [r for r in records if r.start is not None and r.stop is not None and r.stop > r.start + tol]
    if len(win) < 2:
        return None, 0, len(win)
    offs = sorted((e.onset - r.start, k) for k, r in enumerate(win) for e in events
                  if r.start - tol <= e.onset <= r.stop + tol)
    best, best_n, lo = None, 0, 0
    counts: dict[int, int] = {}
    for hi in range(len(offs)):
        counts[offs[hi][1]] = counts.get(offs[hi][1], 0) + 1
        while offs[hi][0] - offs[lo][0] > tol:
            k = offs[lo][1]
            counts[k] -= 1
            if not counts[k]:
                del counts[k]
            lo += 1
        if len(counts) > best_n:
            best_n = len(counts)
            vals = [o for o, _ in offs[lo:hi + 1]]
            best = float(sorted(vals)[len(vals) // 2])
    if best_n < max(2, math.ceil(0.5 * len(win))):
        return None, best_n, len(win)
    return best, best_n, len(win)


def match_records(records: list[Record], events: list[Event], tol: float
                  ) -> tuple[list[tuple[Record, Event]], list[Record], list[tuple[str, object, str]]]:
    """
    Timed records -> channel events, and QC notes.

    Records with a single time take the nearest unmatched train whose onset is within tol.
    Records that give a trial window are matched on their expected onset, window start + the offset
    most windows agree on (window_offset), to the nearest unmatched train within tol; a window with
    no train there is left unmatched (never given the next trial's train), and a window with two
    trains within tol of its expected onset is flagged as ambiguous. When the windows do not agree
    on an offset, each takes the unmatched train inside its window whose duration fits best, then
    the earliest, and that fallback is flagged.
    """
    notes: list[tuple[str, object, str]] = []
    pairs, left = [], []
    taken: set[int] = set()
    off, n_agree, n_win = window_offset(records, events, tol)
    if n_win >= 2:
        if off is not None:
            notes.append(("window_offset", round(off, 6),
                          f"{n_agree} of {n_win} trial-window rows have a train starting {off:.3f} s after the "
                          f"window start (+-{tol:g} s); rows are matched on that expected onset"))
        else:
            notes.append(("window_offset", "n/a",
                          f"the {n_win} trial-window rows do not agree on where the train sits in the window "
                          f"(best: {n_agree}); each row took the best-fitting train inside its window: check "
                          "the parameters against the protocol"))
    for r in sorted(records, key=lambda r: (r.start if r.start is not None else math.inf)):
        if r.start is None:
            left.append(r)
            continue
        is_win = r.stop is not None and r.stop > r.start + tol
        if is_win and off is not None:
            r.expected = r.start + off
        target = r.expected if r.expected is not None else r.start
        best, best_key, close = None, None, 0
        dur = r.params.get("train_duration_s")
        for i, e in enumerate(events):
            if i in taken:
                continue
            if is_win and off is None:
                if not (r.start - tol <= e.onset <= r.stop + tol):
                    continue
                inside = e.end <= r.stop + tol
                fit = abs((e.duration or 0) - dur) if dur is not None else 0.0
                key = (0 if inside else 1, round(fit, 3), e.onset)
                close += 1
            else:
                d = abs(e.onset - target)
                if d > tol:
                    continue
                close += 1
                key = (0, round(d, 6), e.onset)
            if best_key is None or key < best_key:
                best, best_key = i, key
        if best is None:
            left.append(r)
            continue
        if close > 1:
            notes.append(("ambiguous_match", close,
                          f"{r.origin}: {close} unmatched trains "
                          + (f"within {tol:g} s of its expected onset {target:.3f} s" if not (is_win and off is None)
                             else "inside its window")
                          + f"; took the one at {events[best].onset:.3f} s"))
        taken.add(best)
        pairs.append((r, events[best]))
    return pairs, left, notes


def event_from_record(r: Record) -> Event:
    """An event placed by the record's own times (no channel train matched it)."""
    dur = r.params.get("train_duration_s")
    if dur is None and r.stop is not None and r.start is not None and r.stop > r.start:
        dur = r.stop - r.start
    onset = r.expected if r.expected is not None else float(r.start or 0.0)
    if r.expected is not None and r.params.get("train_duration_s") is None:
        dur = None
    ev = Event(onset=onset, duration=dur, timing=r.source)
    ev.add(r.source, r.params, r.extras, r.origin)
    return ev


# --------------------------------------------------------------------------- QC comparisons

COMPARE = {  # parameter: (relative tolerance, absolute tolerance)
    "frequency_hz": (0.10, 0.2),
    "pulse_width_us": (0.20, 20.0),
    "train_duration_s": (0.10, 0.5),
    "n_pulses": (0.10, 1.0),
    "amplitude": (0.10, 0.0),
}


def disagreements(ev: Event) -> list[tuple[str, float, float, str, str]]:
    """(parameter, higher-ranked value, channel value, higher source, channel source) where they differ."""
    out = []
    ch_layers = [(s, p) for s, p in ev.layers if s in CHANNEL_SOURCES]
    hi_layers = [(s, p) for s, p in ev.layers if RANK.get(s, 9) < 2]
    for name, (rel, ab) in COMPARE.items():
        for hs, hp in hi_layers:
            hv = hp.get(name)
            if not isinstance(hv, (int, float)):
                continue
            for cs, cp in ch_layers:
                cv = cp.get(name)
                if not isinstance(cv, (int, float)):
                    continue
                if name == "amplitude" and hp.get("amplitude_unit") != cp.get("amplitude_unit"):
                    continue
                if abs(hv - cv) > max(rel * abs(hv), ab):
                    out.append((name, float(hv), float(cv), hs, cs))
    return out
