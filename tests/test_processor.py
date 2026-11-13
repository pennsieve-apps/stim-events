"""The synthetic fixture through the processor (no Docker): the planted trains are recovered."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from processor import main as M
from processor import params as P
from tests import make_fixture

WRITERS = ["h5py"]
try:
    import pynwb  # noqa: F401
    WRITERS.insert(0, "pynwb")
except ImportError:
    pass


def tsv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run(monkeypatch, input_dir: Path, output_dir: Path, **env) -> int:
    for k in M.DEFAULTS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return M.main()


@pytest.fixture(scope="session", params=WRITERS)
def fixture(request, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp(f"fixture-{request.param}")
    make_fixture.main(d, request.param)
    return d


@pytest.fixture(scope="session")
def out(fixture, tmp_path_factory) -> Path:
    o = tmp_path_factory.mktemp("out")
    mp = pytest.MonkeyPatch()
    try:
        assert run(mp, fixture, o) == 0
    finally:
        mp.undo()
    return o / "stim-events"


@pytest.fixture(scope="session")
def truth(fixture) -> dict:
    return json.loads((fixture / "truth.json").read_text())


def ev(out, rid, trial_type=None):
    return [r for r in tsv(out / "events.tsv") if r["recording_id"] == rid
            and (trial_type is None or r["trial_type"] == trial_type)]


def f(v):
    return None if v == "n/a" else float(v)


def test_layout_and_columns(out):
    for name in ("events.tsv", "stim_summary.csv", "qc.csv", "features.csv", "summary.json", "provenance.json"):
        assert (out / name).is_file(), name
    header = (out / "events.tsv").read_text().splitlines()[0].split("\t")
    assert header[:len(P.EVENT_COLUMNS)] == P.EVENT_COLUMNS
    prov = json.loads((out / "provenance.json").read_text())
    assert set(prov) == {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
                         "wall_seconds", "peak_rss_mb", "threads", "recordings", "citations", "licence_notice"}
    assert all(r["status"] == "ok" for r in prov["recordings"].values())
    summ = json.loads((out / "summary.json").read_text())
    assert len(summ["figures"]) == 6 and all((out / fg["path"]).is_file() for fg in summ["figures"])
    assert all(fg["caption"] for fg in summ["figures"])


def test_trigger_and_table(out, truth):
    rid = "sub-H01_rec-grid"
    got = ev(out, rid, "stimulation")
    want = truth[rid]["events"]
    assert len(got) == len(want) == 10
    for g, w in zip(got, want):
        assert abs(f(g["onset"]) - w["onset"]) < 1e-3          # timing from the line, not the late log
        assert abs(f(g["duration"]) - w["duration"]) < 1e-3
        assert f(g["amplitude"]) == w["amplitude"] and g["amplitude_unit"] == "mA"
        assert int(g["n_pulses"]) == w["n_pulses"]
        assert f(g["pulse_width_us"]) == 250                    # only the table knows it
        assert g["source"] == "table|trigger"
        assert g["site"] == "cervical vagus left"
    # row 7: the table says 10 Hz, the line 20 Hz -> table wins, QC says so
    assert f(got[6]["frequency_hz"]) == 10
    assert [f(g["frequency_hz"]) for i, g in enumerate(got) if i != 6] == \
        [w["frequency_hz"] for i, w in enumerate(want) if i != 6]
    qc = [q for q in rows(out / "qc.csv") if q["recording_id"] == rid and q["check"] == "table_vs_channel"]
    assert len(qc) == 1 and qc[0]["value"] == "frequency_hz" and "stim-0007" in qc[0]["note"]


def test_trial_by_trial_hdf5_attrs(out, truth):
    rid = "sub-R01_rec-trials"
    got = ev(out, rid, "stimulation")
    want = truth[rid]["events"]
    assert len(got) == 6
    for g, w in zip(got, want):
        assert abs(f(g["onset"]) - w["onset"]) < 2e-4          # the train inside the trial window, not its start
        assert abs(f(g["duration"]) - w["duration"]) < 2e-3
        assert f(g["amplitude"]) == w["amplitude"] and g["amplitude_unit"] == "uA"
        assert f(g["frequency_hz"]) == w["frequency_hz"] and int(g["n_pulses"]) == w["n_pulses"]
        assert g["polarity"] == "cathodic" and g["waveform"] == "biphasic-interphase"
        assert g["source"] == "hdf5-attrs|trigger"
    # no disagreement: the blips give the pulse width (200 us) and rate the attributes state
    assert not [q for q in rows(out / "qc.csv") if q["recording_id"] == rid and q["check"] == "table_vs_channel"]


def test_envelopes_with_stim_map(out, truth):
    rid = "sub-R02_rec-closedloop"
    got = ev(out, rid)
    want = truth[rid]["events"]
    assert [g["trial_type"] for g in got] == [w["trial_type"] for w in want]
    for g, w in zip(got, want):
        assert abs(f(g["onset"]) - w["onset"]) < 0.01
        assert abs(f(g["duration"]) - w["duration"]) < 0.05
        assert abs(f(g["amplitude"]) - w["amplitude"]) < 0.2
        assert g["amplitude_unit"] == "n/a"                      # uncalibrated envelope
        assert g["source"] == "envelope"
    assert got[2]["waveform"] == "kilohertz" and got[2]["event_id"] == "block-0001"
    checks = {q["check"] for q in rows(out / "qc.csv") if q["recording_id"] == rid}
    assert {"overlap", "amplitude_unit_unknown"} <= checks


def test_markers(out, truth):
    rid = "sub-H02_rec-markers"
    got = ev(out, rid)
    assert [g["trial_type"] for g in got] == ["baseline", "stimulation", "stimulation"]
    for g, w in zip(got[1:], truth[rid]["events"][1:]):
        assert f(g["onset"]) == w["onset"] and f(g["duration"]) == w["duration"]
        assert f(g["amplitude"]) == w["amplitude"] and f(g["frequency_hz"]) == 20 and f(g["pulse_width_us"]) == 250
        assert g["source"] == "marker" and g["label"].startswith("VNS on")
    assert not any("door" in g["label"] for g in got)


def test_monitor(out, truth):
    rid = "sub-H03_rec-monitor"
    got = ev(out, rid, "stimulation")
    assert len(got) == 3
    for g, w in zip(got, truth[rid]["events"]):
        assert abs(f(g["onset"]) - w["onset"]) < 1e-3
        assert g["amplitude_unit"] == "uA" and abs(f(g["amplitude"]) - 800) < 20
        assert abs(f(g["frequency_hz"]) - 30) < 0.1 and int(g["n_pulses"]) == 60
        assert abs(f(g["pulse_width_us"]) - 200) <= 50
        assert g["polarity"] == "cathodic" and g["waveform"] == "biphasic-interphase"


def test_filename_condition(out):
    got = ev(out, "sub-H04_rec-1mA")
    assert len(got) == 1 and got[0]["trial_type"] == "condition" and got[0]["source"] == "filename"
    assert f(got[0]["amplitude"]) == 1 and got[0]["amplitude_unit"] == "mA"
    assert f(got[0]["frequency_hz"]) == 20 and f(got[0]["pulse_width_us"]) == 200
    assert f(got[0]["duration"]) == 60


def test_summary_and_features(out):
    s = {r["recording_id"]: r for r in rows(out / "stim_summary.csv")}
    assert s["sub-H01_rec-grid"]["parameters_varied"] == "amplitude|frequency_hz"
    assert s["sub-R01_rec-trials"]["amplitude_min"] == "100" and s["sub-R01_rec-trials"]["amplitude_max"] == "600"
    feats = rows(out / "features.csv")
    assert any(r["feature"] == "stim_n_trains" and r["value"] == "10" for r in feats
               if r["recording_id"] == "sub-H01_rec-grid")


def test_name_pattern_condition(fixture, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture, tmp_path, NAME_PATTERN=r"(?P<condition>gain\d+)", FIGURES="false") == 0
    got = ev(tmp_path / "stim-events", "sub-R02_rec-closedloop")
    assert {g["condition"] for g in got} == {"gain05"}
    assert all("filename" in g["source"] for g in got)


def test_marker_pattern_named_groups(fixture, tmp_path, monkeypatch):
    rx = r"VNS on (?P<amplitude_mA>[\d.]+)mA (?P<frequency_hz>\d+)Hz"
    assert run(monkeypatch, fixture, tmp_path, MARKER_PATTERN=rx, FIGURES="false") == 0
    got = ev(tmp_path / "stim-events", "sub-H02_rec-markers", "stimulation")
    assert [f(g["amplitude"]) for g in got] == [1.0, 2.0] and all(g["pulse_width_us"] == "n/a" for g in got)


def test_merged_layout(fixture, tmp_path, monkeypatch):
    """physio and params merged side by side at the root of INPUT_DIR."""
    flat = tmp_path / "in"
    shutil.copytree(fixture / "physio", flat)
    for p in (fixture / "params").iterdir():
        shutil.copy(p, flat / p.name)
    assert run(monkeypatch, flat, tmp_path / "o", FIGURES="false") == 0
    got = ev(tmp_path / "o" / "stim-events", "sub-H01_rec-grid")
    assert len(got) == 10 and all(g["amplitude_unit"] == "mA" for g in got)


def test_without_params_trigger_only(fixture, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture / "physio", tmp_path, FIGURES="false") == 0
    got = ev(tmp_path / "stim-events", "sub-H01_rec-grid")
    assert [f(g["frequency_hz"]) for g in got] == [10] * 5 + [20] * 5
    assert all(g["amplitude"] == "n/a" and g["source"] == "trigger" for g in got)
    # without the stim-map, CtrlAmp's negative values are written as they are, with a QC note
    rc = ev(tmp_path / "stim-events", "sub-R02_rec-closedloop")
    assert any(f(g["amplitude"]) < 0 for g in rc)
    assert any(q["check"] == "envelope_negative" for q in rows(tmp_path / "stim-events" / "qc.csv"))


def test_no_stimulation_is_skipped(tmp_path, monkeypatch):
    d = tmp_path / "fx"
    make_fixture.main(d, "h5py")
    keep = "sub-H04_rec-1mA"
    lines = (d / "physio" / "recordings.csv").read_text().splitlines()
    (d / "physio" / "recordings.csv").write_text("\n".join([lines[0]] + [ln for ln in lines if ln.startswith(keep)]))
    for sub in (d / "physio").glob("sub-*"):
        if sub.name != "sub-H04":
            shutil.rmtree(sub)
    assert run(monkeypatch, d / "physio", tmp_path / "o", NAME_PATTERN="none") == 0
    o = tmp_path / "o" / "stim-events"
    assert len((o / "events.tsv").read_text().splitlines()) == 1
    prov = json.loads((o / "provenance.json").read_text())
    assert prov["recordings"][keep]["status"] == "skipped" and prov["recordings"][keep]["reason"]


def test_exit_codes(tmp_path, monkeypatch, fixture):
    (tmp_path / "empty").mkdir()
    assert run(monkeypatch, tmp_path / "empty", tmp_path / "o1") == 1
    assert run(monkeypatch, fixture, tmp_path / "o2", TRAIN_GAP_S="-1") == 2
    assert run(monkeypatch, fixture, tmp_path / "o3", NAME_PATTERN="(unclosed") == 2
    bad = tmp_path / "bad"
    shutil.copytree(fixture, bad)
    (bad / "params" / "stim-map.yaml").write_text("channels:\n  - match: x\n    kind: nonsense\n")
    assert run(monkeypatch, bad, tmp_path / "o4") == 2


@pytest.mark.parametrize("name,role,unit", [
    ("current_uA", "amplitude", "uA"), ("Amplitude (mA)", "amplitude", "mA"), ("frequency_Hz", "frequency_hz", "Hz"),
    ("PW (us)", "pulse_width_us", "us"), ("pulse_width_ms", "pulse_width_us", "ms"), ("duration_s", "train_duration_s", "s"),
    ("stim_shape", "waveform", None), ("onset_s", "onset", "s"), ("File", "recording", None), ("Rate (kHz)", "frequency_hz", "kHz"),
    ("whatever", "extra", None)])
def test_classify_column(name, role, unit):
    r = P.classify_column(name)
    assert r.role == role and (unit is None or r.unit == unit)


def test_tokens():
    t = P.parse_tokens("100uA_200us_30Hz")
    assert t == {"amplitude": 100.0, "amplitude_unit": "uA", "pulse_width_us": 200.0, "frequency_hz": 30.0}
    assert P.parse_tokens("VNS 1.5 mA 20 Hz PW=0.25ms cathodic")["pulse_width_us"] == 250.0
    assert P.parse_tokens("sub-01_ses-2_rec-1") == {}
    assert P.norm_waveform("biphasicwithdeadzone") == "biphasic-interphase"
    assert P.norm_polarity("Cathodic-first") == "cathodic"


def _write_dataset(root: Path, rec, extra_channel_cols: dict | None = None) -> None:
    """One recording written as physio-import would (h5py writer), with recordings.csv and channels.csv."""
    (root / "sub-X").mkdir(parents=True)
    make_fixture.write_h5py(rec, root / "sub-X" / f"{rec.rid}.nwb")
    with open(root / "recordings.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["recording_id", "subject", "path", "species", "duration_s", "source_format", "source_path"])
        w.writerow([rec.rid, rec.subject, f"sub-X/{rec.rid}.nwb", rec.species, round(rec.duration(), 3),
                    rec.source_format, rec.source_path])
    rows = [dict(c, **(extra_channel_cols or {}).get(c["channel"], {})) for c in rec.channels]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(root / "channels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def _trials_rec(missing: int | None, derived: bool = False):
    """8 trials every 12 s, 25 s windows (they overlap), a 10 Hz 5 s train 10 s into each window."""
    import numpy as np
    starts = [12.0 * k for k in range(8)]
    amps = [100 * (k + 1) for k in range(8)]
    fs = 2000.0
    n = int((starts[-1] + 25.0) * fs)
    ttl = np.zeros(n, dtype=np.int16)
    for k, s in enumerate(starts):
        if k == missing:
            continue
        for p in range(50):
            ttl[int(round((s + 10.0 + p / 10.0) * fs))] = 1
    r = make_fixture.Rec("sub-X_rec-win", "X", "pig", "Sus scrofa", "primary/sub-X/x.hdf5", "hdf5")
    r.add("stim", ttl * 5, "V", fs, typ="stim", original="stim_timing")
    if derived:
        r.add("stim-2", ttl * 5, "V", fs, typ="stim", original="stim_timing processed")
    r.intervals["stimulation_source"] = ("per-trial parameters", {
        "start_time": starts, "stop_time": [s + 25.0 for s in starts], "current_uA": amps,
        "frequency_Hz": [10.0] * 8, "duration_s": [5.0] * 8})
    return r, starts, amps


def test_window_rows_with_a_missing_train(tmp_path, monkeypatch):
    """A failed stimulation must not shift the later rows' parameters onto the wrong trains."""
    rec, starts, amps = _trials_rec(missing=2)
    _write_dataset(tmp_path / "in", rec)
    assert run(monkeypatch, tmp_path / "in", tmp_path / "o", FIGURES="false") == 0
    out = tmp_path / "o" / "stim-events"
    got = ev(out, "sub-X_rec-win", "stimulation")
    assert len(got) == 8
    for g, s, a in zip(got, starts, amps):
        assert abs(f(g["onset"]) - (s + 10.0)) < 1e-3
        assert f(g["amplitude"]) == a
    assert got[2]["source"] == "hdf5-attrs"                    # the row without a train, at its expected onset
    assert all(g["source"] == "hdf5-attrs|trigger" for i, g in enumerate(got) if i != 2)
    qc = rows(out / "qc.csv")
    assert any(q["check"] == "unmatched_row" and "row 2" in q["note"] for q in qc)
    off = [q for q in qc if q["check"] == "window_offset"]
    assert off and abs(float(off[0]["value"]) - 10.0) < 1e-3
    # fixed decimals: millisecond-and-better resolution whatever the size of the number
    line = (out / "events.tsv").read_text().splitlines()[1].split("\t")
    assert line[1] == "10.000000" and line[2] == "5.000000"


def test_derived_stim_copy_is_ignored(tmp_path, monkeypatch):
    rec, starts, _ = _trials_rec(missing=None, derived=True)
    _write_dataset(tmp_path / "in", rec, {"stim": {"derived_from": ""}, "stim-2": {"derived_from": "stim"}})
    assert run(monkeypatch, tmp_path / "in", tmp_path / "o", FIGURES="false") == 0
    got = ev(tmp_path / "o" / "stim-events", "sub-X_rec-win", "stimulation")
    assert len(got) == len(starts)


def test_bare_pulse_is_not_a_stimulus_line(tmp_path, monkeypatch):
    """Without channels.csv a channel named 'pulse' (a finger-pulse transducer) is not read as stim."""
    import numpy as np
    fs = 100.0
    t = np.arange(int(60 * fs)) / fs
    r = make_fixture.Rec("sub-P_rec-1", "P", "human", "Homo sapiens", "raw/p.acq", "acq")
    r.add("pulse", (np.sin(2 * np.pi * 1.2 * t) > 0.9).astype(np.float32), "a.u.", fs, typ="ppg", original="Pulse")
    (tmp_path / "in").mkdir()
    make_fixture.write_h5py(r, tmp_path / "in" / "p.nwb")
    assert run(monkeypatch, tmp_path / "in", tmp_path / "o", FIGURES="false", NAME_PATTERN="none") == 0
    assert len((tmp_path / "o" / "stim-events" / "events.tsv").read_text().splitlines()) == 1
