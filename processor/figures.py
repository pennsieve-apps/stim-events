"""The per-recording timeline figure."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

RESEARCH = "Research output, not a clinical finding."
TYPE_COLOURS = {"block": "#e8e8e8", "baseline": "#9e9e9e", "condition": "#c9d7e8"}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def timeline(path: Path, rid: str, rows: list[dict], varied: list[str], rec_end: float | None,
             overlaps: set[str]) -> None:
    """
    One bar per train at its time in the recording. Vertical position: the second parameter that
    varies (frequency when it varies, else one lane per source channel); colour: amplitude.
    Blocks (kilohertz or other inhibition) in purple, baseline epochs in grey, overlapping trains
    outlined in red.
    """
    stim = [r for r in rows if r["trial_type"] in ("stimulation", "block", "baseline", "condition")]
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
    amps = [_num(r["amplitude"]) for r in stim if r["trial_type"] == "stimulation"]
    amps = [abs(a) for a in amps if a is not None]
    unit = next((r["amplitude_unit"] for r in stim if r["amplitude_unit"] not in ("n/a", "")), "")
    norm = mcolors.Normalize(vmin=min(amps), vmax=max(amps) if max(amps) > min(amps) else min(amps) + 1) if amps \
        else None
    cmap = plt.get_cmap("viridis")
    lane_key = "frequency_hz" if "frequency_hz" in varied else None
    if lane_key:
        levels = sorted({_num(r[lane_key]) for r in stim if _num(r[lane_key]) is not None})
        lanes = {v: i for i, v in enumerate(levels)}
        lane_label = [f"{v:g} Hz" for v in levels]
    else:
        keys = sorted({r["source"].split("|")[0] + (":" + r.get("_channel", "") if r.get("_channel") else "")
                       for r in stim})
        lanes = {k: i for i, k in enumerate(keys)}
        lane_label = keys
    n_lanes = max(1, len(lanes))
    extra_lane = n_lanes   # rows without a lane value (e.g. no frequency)
    used_extra = False
    for r in stim:
        on = _num(r["onset"]) or 0.0
        dur = _num(r["duration"])
        if dur is None or dur <= 0:
            dur = max((rec_end or on + 1) * 0.002, 0.05)
        if lane_key:
            lv = _num(r[lane_key])
            y = lanes.get(lv) if lv is not None else None
        else:
            y = lanes.get(r["source"].split("|")[0] + (":" + r.get("_channel", "") if r.get("_channel") else ""))
        if y is None:
            y, used_extra = extra_lane, True
        a = _num(r["amplitude"])
        if r["trial_type"] == "stimulation" and a is not None and norm is not None:
            col = cmap(norm(abs(a)))
        elif r["trial_type"] == "stimulation":
            col = "#4c72b0"
        else:
            col = TYPE_COLOURS.get(r["trial_type"], "#4c72b0")
        edge = "#d62728" if r["event_id"] in overlaps else "none"
        ax.broken_barh([(on, dur)], (y - 0.38, 0.76), facecolors=col, edgecolors=edge,
                       linewidth=1.5 if edge != "none" else 0, alpha=0.95 if r["trial_type"] != "condition" else 0.5,
                       hatch="///" if r["trial_type"] == "block" else None)
    ticks = list(range(n_lanes)) + ([extra_lane] if used_extra else [])
    labels = lane_label + (["no " + (lane_key or "lane")] if used_extra else [])
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels if ticks else [], fontsize=8)
    ax.set_ylim(-0.7, max(ticks or [0]) + 0.7)
    if rec_end:
        ax.set_xlim(0, rec_end)
    ax.set_xlabel("time from the recording start (s)")
    ax.set_ylabel("frequency" if lane_key else "source")
    n_st = sum(1 for r in stim if r["trial_type"] == "stimulation")
    n_bl = sum(1 for r in stim if r["trial_type"] == "block")
    ttl = f"{rid}: {n_st} stimulation train(s)" + (f", {n_bl} block epoch(s)" if n_bl else "")
    if varied:
        ttl += " - varied: " + ", ".join(varied)
    ax.set_title(ttl, fontsize=10)
    if norm is not None and len(amps):
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, pad=0.01)
        cb.set_label(f"amplitude ({unit})" if unit and unit != "n/a" else "amplitude (unit unknown)", fontsize=8)
    if n_bl:
        from matplotlib.patches import Patch  # noqa: PLC0415
        ax.legend(handles=[Patch(facecolor=TYPE_COLOURS["block"], hatch="///", label="block (e.g. kilohertz)")],
                  loc="upper left", fontsize=7, framealpha=0.8)
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.01, 0.01, RESEARCH, fontsize=7, color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
