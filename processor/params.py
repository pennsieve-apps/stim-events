"""
Stimulation parameters: canonical names, units and values.

Every source (an NWB `stimulation_source` table, a user parameter table, marker text, file names,
SDS metadata) names its parameters its own way: `current_uA`, `Amplitude (mA)`, `I`, `freq`,
`PW (us)`, `stim_shape`. This module maps them onto the events.tsv columns of conventions.md,
converts units (amplitude to uA / mA / V, pulse width to us, durations to s) and normalises the
categorical values (polarity, waveform, trial_type). Names it does not recognise are kept as extra
columns in lowercase snake_case, as conventions.md asks.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# events.tsv columns, in order (conventions.md)
EVENT_COLUMNS = [
    "recording_id", "onset", "duration", "trial_type", "event_id", "amplitude", "amplitude_unit",
    "pulse_width_us", "frequency_hz", "train_duration_s", "train_on_s", "train_off_s", "n_pulses",
    "polarity", "waveform", "site", "contact", "condition", "label", "source",
]
# the stimulation parameters proper (what a source can set on an event)
PARAMS = ["amplitude", "amplitude_unit", "pulse_width_us", "frequency_hz", "train_duration_s", "train_on_s",
          "train_off_s", "n_pulses", "polarity", "waveform", "site", "contact", "condition", "label", "trial_type"]
NUMERIC = {"amplitude", "pulse_width_us", "frequency_hz", "train_duration_s", "train_on_s", "train_off_s",
           "n_pulses"}
TRIAL_TYPES = ("stimulation", "baseline", "block", "marker", "annotation", "artifact", "condition")
WAVEFORMS = ("monophasic", "biphasic", "biphasic-interphase", "kilohertz", "other")
AMP_UNITS = ("uA", "mA", "V")
SOURCES = ("trigger", "monitor", "envelope", "marker", "table", "filename", "sds", "nwb", "hdf5-attrs", "user")

# Column names that say where an event is or which recording a table row belongs to (not parameters).
TIME_KEYS = {
    "onset": ("onset", "onset_s", "start", "start_s", "start_time", "starttime", "time", "time_s", "t", "t_s",
              "stim_start", "stim_onset", "begin"),
    "stop": ("stop", "stop_s", "stop_time", "end", "end_s", "end_time", "offset_s", "stim_end", "stim_stop"),
}
RECORDING_KEYS = ("recording_id", "recording", "file", "filename", "file_name", "source_path", "path", "subject",
                  "subject_id", "session", "session_id", "sample", "sample_id", "performance", "performance_id")

# Canonical parameter -> accepted name stems (after snake_casing and removing a unit suffix).
SYNONYMS = {
    "amplitude": ("amplitude", "amp", "current", "stim_current", "stimulation_current", "stim_amplitude",
                  "stimulus_amplitude", "intensity", "stim_intensity", "i", "ist", "i_stim", "level", "voltage",
                  "stim_voltage", "dose"),
    "frequency_hz": ("frequency", "freq", "frequency_hz", "pulse_frequency", "stim_frequency", "pulse_rate", "rate",
                     "prf", "stim_rate", "repetition_rate", "f"),
    "pulse_width_us": ("pulse_width", "pulsewidth", "pw", "phase_width", "phase_duration", "pulse_duration",
                       "width"),
    "train_duration_s": ("duration", "train_duration", "traindur", "train_dur", "stim_duration",
                         "stimulation_duration", "train_length"),
    "train_on_s": ("train_on", "on_time", "duty_on", "on", "time_on", "cycle_on"),
    "train_off_s": ("train_off", "off_time", "duty_off", "off", "time_off", "cycle_off"),
    "n_pulses": ("n_pulses", "pulses", "num_pulses", "number_of_pulses", "pulse_count", "npulses"),
    "polarity": ("polarity", "first_phase", "leading_phase"),
    "waveform": ("waveform", "shape", "stim_shape", "pulse_shape", "pulse_type", "stim_type", "wave"),
    "site": ("site", "location", "stim_site", "stimulation_site", "electrode", "stim_electrode", "nerve", "target",
             "stim_location"),
    "contact": ("contact", "contacts", "configuration", "config", "electrode_configuration", "montage", "pair"),
    "condition": ("condition", "group", "block_label", "protocol", "setting"),
    "label": ("label", "comment", "comments", "note", "notes", "marker", "annotation", "description", "text"),
    "trial_type": ("trial_type", "event_type", "type", "kind"),
}
# unit suffixes recognised on column names: suffix -> (quantity, unit)
UNIT_SUFFIX = {
    "ua": ("current", "uA"), "µa": ("current", "uA"), "μa": ("current", "uA"), "microamp": ("current", "uA"),
    "microamps": ("current", "uA"), "ma": ("current", "mA"), "milliamp": ("current", "mA"),
    "milliamps": ("current", "mA"), "a": ("current", "A"), "amp": ("current", "A"), "na": ("current", "nA"),
    "v": ("voltage", "V"), "mv": ("voltage", "mV"), "uv": ("voltage", "uV"), "µv": ("voltage", "uV"),
    "volt": ("voltage", "V"), "volts": ("voltage", "V"),
    "hz": ("frequency", "Hz"), "khz": ("frequency", "kHz"),
    "us": ("time", "us"), "µs": ("time", "us"), "μs": ("time", "us"), "usec": ("time", "us"),
    "ms": ("time", "ms"), "msec": ("time", "ms"), "s": ("time", "s"), "sec": ("time", "s"), "secs": ("time", "s"),
    "seconds": ("time", "s"), "min": ("time", "min"),
}
TIME_FACTORS_S = {"us": 1e-6, "ms": 1e-3, "s": 1.0, "min": 60.0}


def snake(name: str) -> str:
    """'Amplitude (mA)' -> 'amplitude_ma'; 'frequency_Hz' -> 'frequency_hz'; 'PW [µs]' -> 'pw_µs'."""
    s = str(name).strip()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s) if not s.isupper() else s
    s = s.lower()
    s = re.sub(r"[^\w µμ]+", "_", s)
    s = re.sub(r"[\s_]+", "_", s).strip("_")
    return s


def split_unit(key: str) -> tuple[str, str | None]:
    """'current_ua' -> ('current', 'uA'); 'pulse_width_us' -> ('pulse_width', 'us')."""
    parts = key.split("_")
    if len(parts) >= 2 and parts[-1] in UNIT_SUFFIX:
        return "_".join(parts[:-1]), UNIT_SUFFIX[parts[-1]][1]
    return key, None


@dataclass
class ColumnRole:
    """What a column of a source table means."""
    original: str
    role: str            # a canonical parameter, 'onset', 'stop', 'recording', 'extra'
    unit: str | None     # unit stated by the column name, e.g. 'uA', 'ms'
    key: str             # canonical name (role for parameters, snake_case for extras)


def classify_column(name: str) -> ColumnRole:
    k = snake(name)
    # snake() splits camelCase, which also splits unit suffixes: "current_uA" -> "current_u_a"; undo that
    k = re.sub(r"_(u|m|n|k)_(a|v|hz)$", r"_\1\2", k)
    stem, unit = split_unit(k)
    for role, names in TIME_KEYS.items():
        if k in names or stem in names:
            return ColumnRole(name, role, unit, role)
    if k in RECORDING_KEYS or stem in RECORDING_KEYS:
        return ColumnRole(name, "recording", None, k if k in RECORDING_KEYS else stem)
    if k in PARAMS:
        return ColumnRole(name, k, _default_unit(k), k)
    for canon, names in SYNONYMS.items():
        if k in names or stem in names:
            if canon == "amplitude" and unit is None and stem in ("voltage", "stim_voltage"):
                unit = "V"
            return ColumnRole(name, canon, unit or _default_unit(canon), canon)
    return ColumnRole(name, "extra", unit, k)


def _default_unit(canon: str) -> str | None:
    return {"pulse_width_us": "us", "frequency_hz": "Hz", "train_duration_s": "s", "train_on_s": "s",
            "train_off_s": "s"}.get(canon)


def to_float(v) -> float | None:
    """A number from a cell: '1.5', '1,5', ' 2 ', 3 -> float; '', 'n/a', NaN -> None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    s = str(v).strip()
    if s.lower() in ("", "n/a", "na", "nan", "none", "null", "-"):
        return None
    s = s.replace(",", ".") if s.count(",") == 1 and "." not in s else s.replace(",", "")
    m = re.match(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        f = float(m.group(0))
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def amplitude_value(v: float, unit: str | None) -> tuple[float, str]:
    """(value, unit) in uA / mA / V: nA -> uA, A -> mA, mV/uV -> V. Unknown unit -> ('n/a')."""
    u = (unit or "").strip()
    ul = u.lower().replace("µ", "u").replace("μ", "u")
    if ul in ("ua", "microamp", "microamps"):
        return v, "uA"
    if ul in ("ma", "milliamp", "milliamps"):
        return v, "mA"
    if ul == "na":
        return v / 1000.0, "uA"
    if ul in ("a", "amp", "amps", "ampere"):
        return (v * 1e6, "uA") if abs(v) < 1e-3 else (v * 1e3, "mA")
    if ul in ("v", "volt", "volts"):
        return v, "V"
    if ul == "mv":
        return v / 1000.0, "V"
    if ul == "uv":
        return v / 1e6, "V"
    return v, "n/a"


def time_value(v: float, unit: str | None, target: str) -> float:
    """Convert a time from `unit` (us, ms, s, min; None = target) to `target` (us or s)."""
    if unit is None:
        return v
    f = TIME_FACTORS_S.get(unit)
    if f is None:
        return v
    return round(v * f / TIME_FACTORS_S[target], 9)


def freq_value(v: float, unit: str | None) -> float:
    return v * 1000.0 if unit == "kHz" else v


def norm_polarity(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s or s in ("n/a", "na", "nan", "none"):
        return None
    if s.startswith(("cath", "neg", "c-", "-")) or s in ("c", "cf", "cathodic-first", "cathodic first"):
        return "cathodic"
    if s.startswith(("anod", "pos", "a-", "+")) or s in ("a", "af", "anodic-first", "anodic first"):
        return "anodic"
    return None


def norm_waveform(v) -> str | None:
    if v is None:
        return None
    s = re.sub(r"[\s_]+", "", str(v).strip().lower())
    if not s or s in ("n/a", "na", "nan", "none"):
        return None
    if any(w in s for w in ("khz", "khfac", "kilohertz", "block", "hfac")):
        return "kilohertz"
    if "mono" in s:
        return "monophasic"
    if "bi" in s and any(w in s for w in ("deadzone", "interphase", "gap", "delay", "ipd", "ipg")):
        return "biphasic-interphase"
    if "bi" in s or "symmetric" in s or "chargebalanced" in s:
        return "biphasic"
    return "other"


def norm_trial_type(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in TRIAL_TYPES:
        return s
    if s in ("stim", "stimulus", "train", "vns", "on"):
        return "stimulation"
    if s in ("rest", "pre", "prestim", "pre-stim", "control", "sham", "off"):
        return "baseline" if s != "sham" else "condition"
    if s in ("khfac", "kilohertz", "inhibition", "nerve block", "kilohertz block"):
        return "block"
    return None


def clean_text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    s = str(v).strip()
    if not s or s.lower() in ("n/a", "nan", "none", "null"):
        return None
    return s.replace("\t", " ").replace("\n", " ")


def from_record(record: dict, roles: dict[str, ColumnRole] | None = None, amplitude_unit: str | None = None
                ) -> tuple[dict, dict]:
    """
    Canonical parameters and extra columns from one source row {column: value}.
    `amplitude_unit` is used when the amplitude column names no unit (a stim-map.yaml setting).
    """
    params: dict = {}
    extras: dict = {}
    for col, val in record.items():
        role = (roles or {}).get(col) or classify_column(col)
        if role.role in ("onset", "stop", "recording"):
            continue
        if role.role == "extra":
            t = clean_text(val) if not isinstance(val, (int, float)) else val
            if t is not None and not (isinstance(t, float) and not math.isfinite(t)):
                extras[role.key] = t
            continue
        set_param(params, role.role, val, role.unit, amplitude_unit)
    return params, extras


def set_param(params: dict, name: str, val, unit: str | None = None, amplitude_unit: str | None = None) -> None:
    """Set a canonical parameter from a raw value, converting units; ignores empty values."""
    if name == "amplitude":
        f = to_float(val)
        if f is None:
            return
        a, u = amplitude_value(f, unit or amplitude_unit)
        params["amplitude"] = a
        params["amplitude_unit"] = u
    elif name == "amplitude_unit":
        t = clean_text(val)
        if t:
            _, u = amplitude_value(1.0, t)
            params["amplitude_unit"] = u
    elif name == "pulse_width_us":
        f = to_float(val)
        if f is not None:
            params[name] = time_value(f, unit or "us", "us")
    elif name in ("train_duration_s", "train_on_s", "train_off_s"):
        f = to_float(val)
        if f is not None:
            params[name] = time_value(f, unit or "s", "s")
    elif name == "frequency_hz":
        f = to_float(val)
        if f is not None:
            params[name] = freq_value(f, unit)
    elif name == "n_pulses":
        f = to_float(val)
        if f is not None:
            params[name] = int(round(f))
    elif name == "polarity":
        p = norm_polarity(val)
        if p:
            params[name] = p
    elif name == "waveform":
        w = norm_waveform(val)
        if w:
            params[name] = w
    elif name == "trial_type":
        t = norm_trial_type(val)
        if t:
            params[name] = t
    else:
        t = clean_text(val)
        if t is not None:
            params[name] = t


# --------------------------------------------------------------------------- text (markers, file names)

_NUM = r"(\d+(?:[.,p]\d+)?)"
_B = r"(?<![A-Za-z0-9.])"          # token start: not inside a word or number
_E = r"(?![A-Za-z])"               # token end: not followed by a letter
TOKEN_PATTERNS = [
    ("amplitude", re.compile(_B + _NUM + r"\s*(uA|µA|μA|mA|nA|mV|V)" + _E, re.IGNORECASE)),
    ("frequency_hz", re.compile(_B + _NUM + r"\s*(kHz|Hz)" + _E, re.IGNORECASE)),
    ("pulse_width_us", re.compile(_B + _NUM + r"\s*(us|µs|μs|usec)" + _E, re.IGNORECASE)),
    ("pulse_width_us", re.compile(r"(?:pw|pulse.?width)\s*[=:_-]?\s*" + _NUM + r"\s*(ms|us|µs)?" + _E, re.IGNORECASE)),
    ("train_duration_s", re.compile(r"(?:dur(?:ation)?|train)\s*[=:_-]?\s*" + _NUM + r"\s*(ms|s|sec|min)" + _E,
                                    re.IGNORECASE)),
]
POLARITY_TOKEN = re.compile(r"(?<![a-z])(cathodic|anodic)(?![a-z])", re.IGNORECASE)
WAVEFORM_TOKEN = re.compile(r"(?<![a-z])(monophasic|biphasic|khfac|kilohertz|khz block)(?![a-z])", re.IGNORECASE)


def _num(s: str) -> float:
    return float(s.replace(",", ".").replace("p", ".").replace("P", "."))


def parse_tokens(text: str) -> dict:
    """
    Parameters written as value-unit tokens in free text (marker labels, file names):
    '1.5mA_20Hz_250us', 'VNS 500 uA 10 Hz', 'PW=0.2ms', 'cathodic'. Each quantity is taken from its
    first occurrence; a quantity written twice with different values is left out (ambiguous).
    """
    found: dict = {}
    seen: dict = {}
    for name, rx in TOKEN_PATTERNS:
        for m in rx.finditer(text):
            v = _num(m.group(1))
            u = m.group(2) if m.lastindex and m.lastindex >= 2 else None
            p: dict = {}
            if name == "amplitude":
                a, au = amplitude_value(v, u.replace("µ", "u").replace("μ", "u") if u else None)
                p = {"amplitude": a, "amplitude_unit": au}
            elif name == "frequency_hz":
                p = {"frequency_hz": v * 1000.0 if u and u.lower() == "khz" else v}
            elif name == "pulse_width_us":
                uu = (u or "us").lower().replace("µ", "u").replace("μ", "u").replace("usec", "us")
                p = {"pulse_width_us": time_value(v, uu, "us")}
            elif name == "train_duration_s":
                uu = {"sec": "s"}.get((u or "s").lower(), (u or "s").lower())
                p = {"train_duration_s": time_value(v, uu, "s")}
            for k, val in p.items():
                if k in seen and seen[k] != val:
                    found.pop(k, None)
                    found.setdefault("_ambiguous", set()).add(k)
                    continue
                seen[k] = val
                if k not in found.get("_ambiguous", set()):
                    found[k] = val
    found.pop("_ambiguous", None)
    if "amplitude" not in found:
        found.pop("amplitude_unit", None)
    m = POLARITY_TOKEN.search(text)
    if m:
        found["polarity"] = m.group(1).lower()
    m = WAVEFORM_TOKEN.search(text)
    if m:
        found["waveform"] = norm_waveform(m.group(1))
    return found


def parse_named(rx: re.Pattern, text: str) -> dict | None:
    """Named groups of a user regex -> (params, extras) merged dict with unit conversion; None = no match."""
    m = rx.search(text)
    if not m:
        return None
    out: dict = {}
    for k, v in m.groupdict().items():
        if v is None:
            continue
        role = classify_column(k)
        if role.role in ("onset", "stop", "recording"):
            continue
        if role.role == "extra":
            out.setdefault("_extras", {})[role.key] = v
        else:
            set_param(out, role.role, v, role.unit)
    return out


def fmt(v, nd: int = 6) -> str:
    """events.tsv cell: numbers without trailing noise, 'n/a' for missing."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return "n/a"
        if v == int(v) and abs(v) < 1e12:
            return str(int(v))
        return f"{v:.{nd}g}"
    s = str(v)
    return s if s else "n/a"
