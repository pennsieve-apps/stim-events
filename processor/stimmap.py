"""
stim-map.yaml: how to read a lab's stimulation channels and parameter tables (all keys optional).

    version: 1
    channels:                    # stimulation channels, matched in order (first match wins)
      - match: CtrlAmp           # regex (case-insensitive) on the channel name, original name or description
        kind: envelope           # trigger | monitor | envelope | ignore (default: decided from name and content)
        quantity: amplitude      # envelope only: amplitude | frequency_hz | pulse_width_us
        unit: mA                 # unit of the values after `scale` (uA, mA, A, V, mV; Hz; us)
        scale: 1.0               # factor applied to the channel's values
        threshold: 1.0           # in the channel's unit after scale (TRIGGER_THRESHOLD / ENVELOPE_THRESHOLD)
        site: right cervical vagus
        contact: distal electrode
        group: control           # channels with the same group describe one stimulator: events merged
        positive: {trial_type: stimulation}
        negative: {trial_type: block, waveform: kilohertz}   # value becomes |value|
        set: {pulse_width_us: 50, waveform: biphasic}       # fixed parameters of this channel's events
    tables:                      # parameter tables in the params port
      - file: "stim_log*.xlsx"   # glob on the file name or its path inside the params port
        sheet: Sheet1            # XLSX sheet name or index (default: the first)
        columns: {amplitude: "I (mA)", onset: "t_on", frequency_hz: "Rate"}   # our name: their column
        amplitude_unit: mA       # when the amplitude column does not say
        time_unit: s             # onset / stop columns: s, ms or min
        recording_column: File   # which column names the recording (recording_id, file name or subject)
        match: time              # time (rows to trains by time) | order (k-th row = k-th train) |
                                 # recording (one row of parameters for the whole recording)
    defaults: {site: cervical vagus}   # parameters every event gets when no source says otherwise
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from processor import params as P

CHANNEL_KEYS = {"match", "kind", "quantity", "unit", "scale", "threshold", "site", "contact", "group", "positive",
                "negative", "set"}
TABLE_KEYS = {"file", "sheet", "columns", "amplitude_unit", "time_unit", "recording_column", "match"}
TOP_KEYS = {"version", "channels", "tables", "defaults"}
KINDS = ("trigger", "monitor", "envelope", "ignore")
QUANTITIES = ("amplitude", "frequency_hz", "pulse_width_us")
MATCHES = ("time", "order", "recording")


@dataclass
class ChannelRule:
    match: re.Pattern
    kind: str | None = None
    quantity: str = "amplitude"
    unit: str | None = None
    scale: float = 1.0
    threshold: float | None = None
    site: str | None = None
    contact: str | None = None
    group: str | None = None
    positive: dict = field(default_factory=dict)
    negative: dict = field(default_factory=dict)
    set: dict = field(default_factory=dict)


@dataclass
class TableRule:
    file: str
    sheet: object = None
    columns: dict = field(default_factory=dict)
    amplitude_unit: str | None = None
    time_unit: str = "s"
    recording_column: str | None = None
    match: str | None = None


@dataclass
class StimMap:
    channels: list[ChannelRule] = field(default_factory=list)
    tables: list[TableRule] = field(default_factory=list)
    defaults: dict = field(default_factory=dict)
    path: str | None = None

    def channel_rule(self, *texts: str) -> ChannelRule | None:
        for r in self.channels:
            if any(t and r.match.search(t) for t in texts):
                return r
        return None

    def table_rule(self, path: Path, rel: str) -> TableRule | None:
        from fnmatch import fnmatch  # noqa: PLC0415
        for r in self.tables:
            if fnmatch(path.name, r.file) or fnmatch(rel, r.file):
                return r
        return None


def _check_params(where: str, d) -> dict:
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a mapping of parameter: value")
    allowed = set(P.PARAMS) | {"absolute"}
    bad = [k for k in d if k not in allowed]
    if bad:
        raise ValueError(f"{where}: unknown parameter(s) {', '.join(map(str, bad))}; known: {', '.join(P.PARAMS)}")
    return dict(d)


def load(path: Path | None) -> StimMap:
    """Parse and check a stim-map.yaml; raises ValueError with a message a user can act on."""
    if path is None:
        return StimMap()
    import yaml  # noqa: PLC0415

    try:
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{path.name}: not valid YAML ({e})") from None
    if not isinstance(doc, dict):
        raise ValueError(f"{path.name}: the top level must be a mapping")
    bad = set(doc) - TOP_KEYS
    if bad:
        raise ValueError(f"{path.name}: unknown key(s) {', '.join(sorted(bad))}; known: {', '.join(sorted(TOP_KEYS))}")
    sm = StimMap(path=str(path))
    for i, c in enumerate(doc.get("channels") or []):
        where = f"{path.name} channels[{i}]"
        if not isinstance(c, dict) or "match" not in c:
            raise ValueError(f"{where}: needs a `match` regex")
        bad = set(c) - CHANNEL_KEYS
        if bad:
            raise ValueError(f"{where}: unknown key(s) {', '.join(sorted(bad))}")
        try:
            rx = re.compile(str(c["match"]), re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"{where}: match is not a valid regex ({e})") from None
        kind = c.get("kind")
        if kind is not None and kind not in KINDS:
            raise ValueError(f"{where}: kind must be one of {', '.join(KINDS)}")
        q = c.get("quantity", "amplitude")
        if q not in QUANTITIES:
            raise ValueError(f"{where}: quantity must be one of {', '.join(QUANTITIES)}")
        try:
            scale = float(c.get("scale", 1.0))
            thr = None if c.get("threshold") is None else float(c["threshold"])
        except (TypeError, ValueError):
            raise ValueError(f"{where}: scale and threshold must be numbers") from None
        sm.channels.append(ChannelRule(
            match=rx, kind=kind, quantity=q, unit=c.get("unit"), scale=scale, threshold=thr,
            site=c.get("site"), contact=c.get("contact"), group=c.get("group"),
            positive=_check_params(f"{where}.positive", c.get("positive")),
            negative=_check_params(f"{where}.negative", c.get("negative")),
            set=_check_params(f"{where}.set", c.get("set"))))
    for i, t in enumerate(doc.get("tables") or []):
        where = f"{path.name} tables[{i}]"
        if not isinstance(t, dict) or "file" not in t:
            raise ValueError(f"{where}: needs a `file` glob")
        bad = set(t) - TABLE_KEYS
        if bad:
            raise ValueError(f"{where}: unknown key(s) {', '.join(sorted(bad))}")
        cols = t.get("columns") or {}
        if not isinstance(cols, dict):
            raise ValueError(f"{where}: columns must map our names to your column names")
        known = set(P.PARAMS) | {"onset", "stop"}
        badc = [k for k in cols if k not in known]
        if badc:
            raise ValueError(f"{where}: unknown column key(s) {', '.join(badc)}; known: {', '.join(sorted(known))}")
        m = t.get("match")
        if m is not None and m not in MATCHES:
            raise ValueError(f"{where}: match must be one of {', '.join(MATCHES)}")
        tu = t.get("time_unit", "s")
        if tu not in ("s", "ms", "min"):
            raise ValueError(f"{where}: time_unit must be s, ms or min")
        sm.tables.append(TableRule(file=str(t["file"]), sheet=t.get("sheet"), columns=dict(cols),
                                   amplitude_unit=t.get("amplitude_unit"), time_unit=tu,
                                   recording_column=t.get("recording_column"), match=m))
    sm.defaults = _check_params(f"{path.name} defaults", doc.get("defaults"))
    return sm
