"""
Stimulation epochs from stimulation channels, read one slice at a time.

Three kinds of channel (conventions.md, channel type `stim`):

- trigger / TTL: a two-level line, high while (or at the onset of) a pulse. Pulses are found by a
  threshold crossing with hysteresis (60 % / 40 % of the way from the resting to the active level,
  or TRIGGER_THRESHOLD); edges closer than PULSE_MERGE_S belong to one pulse (lines that mark each
  pulse edge with a one-sample blip, as dataset 349's `stim_timing`, give the pulse width as the
  span between the blips); pulses closer than TRAIN_GAP_S form one train. Rate = 1 / median
  inter-pulse interval, n_pulses counted, train duration = last onset - first onset + one interval.
  A pulse group of 3 or more edges at >= 1 kHz is a kilohertz burst (trial_type block).
- monitor: the stimulus current or voltage itself (rests near zero, pulses of either sign). The
  same pulse detection on |x - baseline| (threshold 10 x the noise, at least 2 % of the largest
  pulse) adds the amplitude (median peak per train, in the channel's unit), the polarity (sign of the
  first phase; negative = cathodic) and the waveform (one phase: monophasic; two phases of opposite
  sign: biphasic, with a gap: biphasic-interphase).
- envelope: a stimulation parameter over time (amplitude or frequency set by a stimulator or a
  closed-loop controller, as dataset 655's StimAmp / CtrlAmp). An epoch is where |value| is above
  ENVELOPE_THRESHOLD (default 10 % of the channel's largest value); epochs of the same sign closer
  than ENVELOPE_GAP_S are joined (missing samples count as off), and an epoch is split where the
  1 s median steps by more than ENVELOPE_STEP of its level. Value per epoch = median.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("stim-events")

CHUNK = 1_000_000          # samples read at a time
KHZ_HZ = 1000.0            # pulse rates at or above this are kilohertz (block) waveforms

TRIGGER_WORDS = re.compile(r"ttl|trig|sync|timing|marker|digital|\bdin\b|event|gate|strobe", re.IGNORECASE)
ENVELOPE_WORDS = re.compile(r"amp(?!le)|ampl|envelope|\benv\b|freq|level|setpoint|ctrl|control|command|\bcmd\b|"
                            r"intensity|dose|rate", re.IGNORECASE)
MONITOR_WORDS = re.compile(r"monitor|\bmon\b|isense|imon|vmon|current|voltage|waveform", re.IGNORECASE)
CURRENT_UNITS = {"a", "ma", "ua", "µa", "μa", "na", "v", "mv", "uv", "µv"}


@dataclass
class Settings:
    train_gap_s: float = 2.0
    pulse_merge_s: float = 0.002
    trigger_threshold: float | None = None     # None = auto
    envelope_threshold: float | None = None    # None = auto (10 % of the largest |value|)
    envelope_gap_s: float = 5.0
    envelope_step: float = 0.25


@dataclass
class Epoch:
    """One train (trigger/monitor), kilohertz burst, or envelope epoch, in the channel's own terms."""
    kind: str                          # trigger | monitor | envelope
    channel: str
    onset: float
    end: float
    n_pulses: int | None = None
    frequency_hz: float | None = None
    frequency_min: float | None = None
    frequency_max: float | None = None
    pulse_width_us: float | None = None
    amplitude: float | None = None     # channel unit (monitor peak, envelope median)
    value_min: float | None = None
    value_max: float | None = None
    sign: int = 1
    polarity: str | None = None
    waveform: str | None = None
    khz: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.onset)


@dataclass
class ChannelStats:
    n: int = 0
    n_finite: int = 0
    vmin: float = math.inf
    vmax: float = -math.inf
    medians: list = field(default_factory=list)
    p999: float = 0.0
    mad: list = field(default_factory=list)

    @property
    def median(self) -> float:
        return float(np.median(self.medians)) if self.medians else 0.0


def scan(nf, ch) -> ChannelStats:
    """First pass: range, a robust median, noise and the 99.9th percentile of |x|."""
    st = ChannelStats()
    for i0 in range(0, ch.n, CHUNK):
        _, x = nf.read(ch, i0, min(ch.n, i0 + CHUNK))
        st.n += len(x)
        fin = x[np.isfinite(x)]
        if not len(fin):
            continue
        st.n_finite += len(fin)
        st.vmin = min(st.vmin, float(fin.min()))
        st.vmax = max(st.vmax, float(fin.max()))
        med = float(np.median(fin))
        st.medians.append(med)
        st.mad.append(float(np.median(np.abs(fin - med))))
        st.p999 = max(st.p999, float(np.percentile(np.abs(fin), 99.9)))
    return st


def two_level_fraction(nf, ch, st: ChannelStats, max_samples: int = 2_000_000) -> float:
    """Share of finite samples within 10 % of the range from the minimum or the maximum."""
    rng = st.vmax - st.vmin
    if not math.isfinite(rng) or rng <= 0:
        return 1.0
    hit = tot = 0
    for i0 in range(0, min(ch.n, max_samples), CHUNK):
        _, x = nf.read(ch, i0, min(ch.n, i0 + CHUNK, max_samples))
        x = x[np.isfinite(x)]
        tot += len(x)
        hit += int(np.count_nonzero((x - st.vmin <= 0.1 * rng) | (st.vmax - x <= 0.1 * rng)))
    return hit / tot if tot else 1.0


def classify(name: str, original: str, description: str, unit: str, rate: float | None, nf, ch,
             st: ChannelStats) -> tuple[str, str]:
    """(kind, why): trigger | monitor | envelope from the name, else from the content."""
    text = " ".join([name, original, description])
    if TRIGGER_WORDS.search(text):
        return "trigger", "name"
    if ENVELOPE_WORDS.search(original or "") or ENVELOPE_WORDS.search(description or ""):
        return "envelope", "name"
    if MONITOR_WORDS.search(text):
        return "monitor", "name"
    frac = two_level_fraction(nf, ch, st)
    if frac >= 0.99:
        return "trigger", f"content: {frac:.1%} of samples at two levels"
    if (unit or "").strip().lower() in CURRENT_UNITS and (rate or 0) >= 2000:
        return "monitor", f"content: not two-level, {unit} at {rate:g} Hz"
    return "envelope", f"content: not two-level ({frac:.1%}), slow or without a current/voltage unit"


# --------------------------------------------------------------------------- pulses (trigger and monitor)

def _ffill_state(s: np.ndarray, prev: float) -> np.ndarray:
    """Forward-fill NaN (between the two hysteresis thresholds) with the last state."""
    if np.isnan(s[0]):
        s[0] = prev
    idx = np.where(np.isnan(s), 0, np.arange(len(s)))
    np.maximum.accumulate(idx, out=idx)
    return s[idx]


def find_runs(nf, ch, center: float, th_hi: float, th_lo: float, absolute: bool, invert: bool = False
              ) -> dict[str, np.ndarray]:
    """
    Active runs of a pulse channel: (onset, end, n_samples, peak deviation, sign of the peak) per run.
    absolute: activity is |x - center| (monitor), else (x - center) * (-1 if invert) (trigger).
    Runs are cut at gaps in the time base, so a pulse never spans two trial windows.
    """
    rate = float(ch.rate or 1.0)
    dt = 1.0 / rate
    out = {k: [] for k in ("onset", "end", "n", "peak", "sign", "sign_end")}
    prev_state = 0.0
    open_run = None  # [onset, last_t, n, peak, sign]
    for i0 in range(0, ch.n, CHUNK):
        t, x = nf.read(ch, i0, min(ch.n, i0 + CHUNK))
        dev = x - center
        if invert:
            dev = -dev
        dev[~np.isfinite(dev)] = 0.0
        a = np.abs(dev) if absolute else dev
        s = np.full(len(a), np.nan)
        s[a >= th_hi] = 1.0
        s[a <= th_lo] = 0.0
        # breaks in the time base (trial-by-trial data, overlapping windows): force inactive there
        if ch.has_timestamps and len(t) > 1:
            dtt = np.diff(t)
            brk = np.flatnonzero((dtt > 5 * dt) | (dtt <= 0))
            s[brk] = 0.0
        s = _ffill_state(s, prev_state)
        d = np.diff(np.concatenate(([prev_state], s)))
        starts = np.flatnonzero(d > 0)
        ends = np.flatnonzero(d < 0)       # first inactive sample
        # close a run left open by the previous chunk
        if open_run is not None:
            if len(ends) and (not len(starts) or ends[0] <= starts[0]):
                e = ends[0]
                if e > 0:
                    seg = a[:e]
                    k = int(np.argmax(seg))
                    open_run[3] = max(open_run[3], float(seg[k]))
                    open_run[1] = float(t[e - 1])
                    open_run[2] += e
                se = int(np.sign(dev[e - 1]) or 1) if e > 0 else open_run[4]
                for key, v in zip(("onset", "end", "n", "peak", "sign", "sign_end"),
                                  (open_run[0], open_run[1] + dt, open_run[2], open_run[3], open_run[4], se)):
                    out[key].append(np.array([v]))
                open_run = None
                ends = ends[1:]
            else:  # still open through the whole chunk
                k = int(np.argmax(a))
                open_run[3] = max(open_run[3], float(a[k]))
                open_run[1] = float(t[-1])
                open_run[2] += len(a)
                prev_state = s[-1]
                continue
        n_closed = min(len(starts), len(ends))
        if n_closed:
            st_, en_ = starts[:n_closed], ends[:n_closed]
            bounds = np.empty(2 * n_closed, dtype=np.int64)
            bounds[0::2], bounds[1::2] = st_, en_
            peaks = np.maximum.reduceat(a, bounds)[0::2]
            # sign of the first phase: the deviation where the run starts
            signs = np.sign(dev[st_]).astype(int) if absolute else np.ones(n_closed, dtype=int)
            signs[signs == 0] = 1
            sign_end = np.sign(dev[en_ - 1]).astype(int) if absolute else np.ones(n_closed, dtype=int)
            sign_end[sign_end == 0] = 1
            out["onset"].append(t[st_])
            out["end"].append(t[en_ - 1] + dt)
            out["n"].append((en_ - st_).astype(np.int64))
            out["peak"].append(peaks)
            out["sign"].append(signs)
            out["sign_end"].append(sign_end)
        if len(starts) > n_closed:   # a run still open at the chunk's end
            s0 = starts[n_closed]
            seg = a[s0:]
            k = int(np.argmax(seg))
            open_run = [float(t[s0]), float(t[-1]), len(seg), float(seg[k]), int(np.sign(dev[s0]) or 1)]
        prev_state = s[-1]
    if open_run is not None:
        for key, v in zip(("onset", "end", "n", "peak", "sign", "sign_end"),
                          (open_run[0], open_run[1] + dt, open_run[2], open_run[3], open_run[4], open_run[4])):
            out[key].append(np.array([v]))
    res = {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}
    if len(res["onset"]):
        order = np.argsort(res["onset"], kind="stable")
        res = {k: v[order] for k, v in res.items()}
        # the same pulse seen twice (overlapping trial windows): drop exact repeats
        keep = np.ones(len(order), dtype=bool)
        keep[1:] = np.diff(res["onset"]) > 0.5 * dt
        res = {k: v[keep] for k, v in res.items()}
    return res


def pulses_from_runs(runs: dict, merge_s: float, rate: float, monitor: bool) -> dict[str, np.ndarray]:
    """Group runs closer than merge_s into pulses: onset, end, width_s (one phase), peak, first sign,
    n_runs, waveform code (1 mono, 2 biphasic, 3 biphasic-interphase, 0 unknown), khz flag, khz rate."""
    on, en = runs["onset"], runs["end"]
    if not len(on):
        return {}
    gap = on[1:] - en[:-1]
    newp = np.concatenate(([True], gap > merge_s))
    gid = np.cumsum(newp) - 1
    first = np.flatnonzero(newp)
    last = np.concatenate((first[1:] - 1, [len(on) - 1]))
    n_runs = last - first + 1
    dt = 1.0 / rate
    run_w = en - on
    blip = runs["n"] <= 2
    all_blips = np.minimum.reduceat(blip.astype(int), first).astype(bool)
    width = np.where(all_blips & (n_runs >= 2), on[np.minimum(first + 1, last)] - on[first],
                     np.where(all_blips, np.nan, run_w[first]))
    peak = np.maximum.reduceat(runs["peak"], first)
    sign0 = runs["sign"][first]
    span_on = on[last] - on[first]
    khz = (n_runs >= 3) & (span_on / np.maximum(n_runs - 1, 1) <= 1.0 / KHZ_HZ + 1e-12)
    khz_rate = np.where(khz, (n_runs - 1) / np.maximum(span_on, dt), np.nan)
    wave = np.zeros(len(first), dtype=int)
    if monitor:
        one = n_runs == 1
        wave[one] = np.where(runs["sign_end"][first][one] != sign0[one], 2, 1)
        two = n_runs == 2
        if two.any():
            opp = runs["sign"][np.minimum(first + 1, last)] != sign0
            gap2 = on[np.minimum(first + 1, last)] - en[first]
            # two phases split by at least one sample back near the baseline: an interphase gap
            wave[two & opp & (gap2 < 0.5 * dt)] = 2
            wave[two & opp & (gap2 >= 0.5 * dt)] = 3
    del gid
    return {"onset": on[first], "end": en[last], "width_s": width, "peak": peak, "sign": sign0,
            "n_runs": n_runs, "wave": wave, "khz": khz, "khz_rate": khz_rate}


WAVE_NAMES = {1: "monophasic", 2: "biphasic", 3: "biphasic-interphase"}


def trains_from_pulses(p: dict, kind: str, channel: str, train_gap_s: float) -> list[Epoch]:
    """Pulses -> trains (gap > train_gap_s starts a new one); kilohertz bursts are epochs of their own."""
    if not p:
        return []
    out: list[Epoch] = []
    khz = p["khz"]
    for i in np.flatnonzero(khz):
        out.append(Epoch(kind=kind, channel=channel, onset=float(p["onset"][i]), end=float(p["end"][i]),
                         frequency_hz=float(p["khz_rate"][i]), khz=True, waveform="kilohertz",
                         amplitude=float(p["peak"][i]) if kind == "monitor" else None))
    idx = np.flatnonzero(~khz)
    if len(idx):
        on = p["onset"][idx]
        brk = np.concatenate(([True], np.diff(on) > train_gap_s))
        starts = np.flatnonzero(brk)
        stops = np.concatenate((starts[1:], [len(idx)]))
        for a, b in zip(starts, stops):
            sel = idx[a:b]
            ons = p["onset"][sel]
            n = len(sel)
            e = Epoch(kind=kind, channel=channel, onset=float(ons[0]), end=float(p["end"][sel[-1]]), n_pulses=n)
            if n >= 2:
                ipi = np.diff(ons)
                med = float(np.median(ipi))
                e.frequency_hz = 1.0 / med if med > 0 else None
                e.frequency_min, e.frequency_max = 1.0 / float(ipi.max()), 1.0 / max(float(ipi.min()), 1e-9)
                e.end = float(ons[-1]) + med
                if e.frequency_hz and (e.frequency_max - e.frequency_min) > 0.2 * e.frequency_hz:
                    e.notes.append(f"pulse rate varies within the train ({e.frequency_min:.3g}-{e.frequency_max:.3g} Hz)")
            w = p["width_s"][sel]
            w = w[np.isfinite(w)]
            if len(w):
                e.pulse_width_us = float(np.median(w)) * 1e6
                if n == 1:
                    e.end = max(e.end, e.onset + float(w[0]))
            if kind == "monitor":
                e.amplitude = float(np.median(p["peak"][sel]))
                sg = p["sign"][sel]
                e.polarity = "cathodic" if np.mean(sg < 0) > 0.5 else "anodic"
                codes = p["wave"][sel]
                codes = codes[codes > 0]
                if len(codes):
                    vals, cnt = np.unique(codes, return_counts=True)
                    e.waveform = WAVE_NAMES.get(int(vals[np.argmax(cnt)]))
            out.append(e)
    out.sort(key=lambda e: e.onset)
    return out


def detect_pulses(nf, ch, st: ChannelStats, kind: str, s: Settings) -> tuple[list[Epoch], dict]:
    """Trains of a trigger or monitor channel, and how they were found."""
    info: dict = {}
    if not math.isfinite(st.vmin) or st.vmax - st.vmin <= 0:
        return [], {"note": "flat channel: no pulses"}
    if kind == "trigger":
        med = st.median
        invert = (st.vmax - med) < (med - st.vmin)       # resting high, pulses go low
        lo, hi = (st.vmax, st.vmin) if invert else (st.vmin, st.vmax)
        rng = abs(hi - lo)
        if s.trigger_threshold is not None:
            th = abs(s.trigger_threshold - lo)
            th_hi, th_lo = th, th * 0.67
        else:
            th_hi, th_lo = 0.6 * rng, 0.4 * rng
        info = {"rest_level": lo, "active_level": hi, "threshold": lo + (th_hi if not invert else -th_hi),
                "inverted": bool(invert)}
        runs = find_runs(nf, ch, lo, th_hi, th_lo, absolute=False, invert=invert)
        pulses = pulses_from_runs(runs, s.pulse_merge_s, float(ch.rate or 1.0), monitor=False)
    else:
        base = st.median
        noise = 1.4826 * (float(np.median(st.mad)) if st.mad else 0.0)
        big = max(abs(st.vmax - base), abs(st.vmin - base))
        if s.trigger_threshold is not None:
            th_hi = abs(s.trigger_threshold)
        else:
            th_hi = max(10.0 * noise, 0.02 * big)
        th_lo = 0.67 * th_hi
        info = {"baseline": base, "noise": noise, "threshold": th_hi}
        runs = find_runs(nf, ch, base, th_hi, th_lo, absolute=True)
        pulses = pulses_from_runs(runs, s.pulse_merge_s, float(ch.rate or 1.0), monitor=True)
    info["n_pulses"] = int(len(pulses.get("onset", []))) if pulses else 0
    return trains_from_pulses(pulses, kind, ch.name, s.train_gap_s), info


# --------------------------------------------------------------------------- envelopes

def detect_envelope(nf, ch, st: ChannelStats, s: Settings) -> tuple[list[Epoch], dict]:
    """Epochs where |value| > threshold; 1 s pieces joined into epochs by sign, gap and step."""
    if not math.isfinite(st.vmin):
        return [], {"note": "no finite samples"}
    biggest = max(abs(st.vmin), abs(st.vmax), st.p999)
    thr = s.envelope_threshold if s.envelope_threshold is not None else 0.1 * max(st.p999, 1e-12)
    info = {"threshold": thr, "largest": biggest}
    if biggest <= thr:
        return [], {**info, "note": "never above the threshold"}
    rate = float(ch.rate or 1.0)
    dt = 1.0 / rate
    pieces = []   # (onset, end, sign, median, min, max, n)
    for i0 in range(0, ch.n, CHUNK):
        t, x = nf.read(ch, i0, min(ch.n, i0 + CHUNK))
        on = np.isfinite(x) & (np.abs(np.nan_to_num(x)) > thr)
        if not on.any():
            continue
        sg = np.where(on, np.sign(np.nan_to_num(x)), 0).astype(int)
        # cut at time-base gaps
        if ch.has_timestamps and len(t) > 1:
            dtt = np.diff(t)
            brk = np.flatnonzero((dtt > 5 * dt) | (dtt <= 0))
        else:
            brk = np.array([], dtype=int)
        sec = np.floor((t - t[0])).astype(np.int64)
        key_change = np.zeros(len(x), dtype=bool)
        key_change[0] = True
        key_change[1:] = (sg[1:] != sg[:-1]) | (sec[1:] != sec[:-1])
        key_change[brk + 1] = True
        bnd = np.flatnonzero(key_change)
        ends = np.concatenate((bnd[1:], [len(x)]))
        for a, b in zip(bnd, ends):
            if sg[a] == 0:
                continue
            v = x[a:b]
            pieces.append((float(t[a]), float(t[b - 1]) + dt, int(sg[a]), float(np.median(v)), float(v.min()),
                           float(v.max()), int(b - a)))
    if not pieces:
        return [], {**info, "note": "never above the threshold"}
    pieces.sort(key=lambda p: p[0])
    epochs: list[list] = [[pieces[0]]]
    for p in pieces[1:]:
        cur = epochs[-1]
        prev = cur[-1]
        level = max(abs(prev[3]), abs(p[3]))
        joined = (p[2] == prev[2] and p[0] - prev[1] <= s.envelope_gap_s
                  and abs(p[3] - prev[3]) <= s.envelope_step * level)
        if joined:
            cur.append(p)
        else:
            epochs.append([p])
    out = []
    for ps in epochs:
        n = np.array([p[6] for p in ps], dtype=float)
        med = np.array([p[3] for p in ps])
        order = np.argsort(med)
        cw = np.cumsum(n[order])
        wmed = float(med[order][np.searchsorted(cw, cw[-1] / 2.0)])
        e = Epoch(kind="envelope", channel=ch.name, onset=ps[0][0], end=ps[-1][1], amplitude=wmed,
                  value_min=min(p[4] for p in ps), value_max=max(p[5] for p in ps), sign=ps[0][2])
        covered = float(n.sum()) * dt
        if e.duration > 0 and covered < 0.5 * e.duration:
            e.notes.append(f"value present in {covered / e.duration:.0%} of the epoch (gaps up to "
                           f"{s.envelope_gap_s:g} s joined)")
        if abs(e.value_max - e.value_min) > 0.2 * max(abs(wmed), 1e-12):
            e.notes.append(f"value varies within the epoch ({e.value_min:.4g} to {e.value_max:.4g})")
        out.append(e)
    return out, info
