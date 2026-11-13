# stim-events

**Stimulation Events** finds the stimulation epochs of physiological recordings and their
parameters, and writes them as one BIDS-style `events.tsv` that the later steps (heart rate,
blood pressure, dose-response) use to cut baseline, during and after windows. It is this app's own
code (Apache-2.0), written by Pennsieve for the Cardiovascular Response to Nerve Stimulation
workflow; no published stimulation-epoch tool with an open licence exists to package. It reads
the NWB files written by **physio-import**, which follow the **NWB** standard from the Neurodata
Without Borders project (Oliver Rübel and colleagues, [eLife 2022](https://doi.org/10.7554/eLife.78362)),
with h5py (Andrew Collette and the h5py contributors, <https://github.com/h5py/h5py>), and writes
its table in the `events.tsv` shape of the Brain Imaging Data Structure (Krzysztof Gorgolewski and
colleagues, [Scientific Data 2016](https://doi.org/10.1038/sdata.2016.44)). The SPARC Data Structure
(Anita Bandrowski and colleagues, [bioRxiv 2021](https://doi.org/10.1101/2021.02.10.430563)) is one
optional metadata source. The demo data are Pennsieve Discover datasets 349 (Oliver Armitage and
colleagues, pig vagus nerve stimulation grid, [doi:10.26275/dm5l-tspb](https://doi.org/10.26275/dm5l-tspb))
and 655 (Shane A. Bender and colleagues, closed-loop heart-rate control by vagus stimulation and
kilohertz block in rats, [doi:10.26275/5cwo-mkpz](https://doi.org/10.26275/5cwo-mkpz)).

**For a physiologist:** give it your recordings (after physio-import) and, if you have one, your
stimulation log. It finds every train on the trigger, stimulus monitor or stimulator-setting
channel, reads the amplitude, pulse width, rate, duration, polarity, waveform and site from wherever
they are written down (the recording file itself, your spreadsheet, the marker text, the file name),
and lists one row per train. It tells you when something does not add up: two trains that overlap,
a train that starts too soon after the previous one for a clean baseline, or a spreadsheet row that
says 10 Hz where the trigger line pulses at 20 Hz. A timeline figure per recording shows the trains
coloured by amplitude.

**For a data scientist:** each source becomes a layer of parameters on an event; layers are merged
by a fixed priority (below), timing comes from the channel whenever a table row or marker was
matched to a detected train, and every row says in `source` where its values came from. Channels are
read with h5py in 1-million-sample slices, so a 30 kHz trigger line of hours never sits in memory.

## Inputs

| Port | Placement | What |
|---|---|---|
| `physio` | subfolder, required | physio-import's `physio` output: `recordings.csv`, `channels.csv`, `events.tsv` (markers) and one NWB file per recording. Any NWB files also work: without `channels.csv`, stimulation channels are found by name (`stim*`, TTL, trigger, ...) or under `/stimulus/presentation`. |
| `params` | subfolder, optional | Parameter tables (CSV, TSV, XLSX), a `stim-map.yaml`, SPARC Data Structure `manifest.xlsx` / `performances.xlsx` / `samples.xlsx` / `subjects.xlsx`. |

Files are found by name and content, recursively, following symlinks; the two ports may also be
merged side by side at the root of the input.

### Sources, in priority order

A higher source's value wins; a lower source only fills what is missing.

1. **The NWB `stimulation_source` table** (per-trial parameters physio-import copied from the
   source file, e.g. dataset 349's HDF5 trial attributes `current_uA`, `frequency_Hz`,
   `pulse_width_us`, `duration_s`, `polarity`, `stim_shape`; `source` = `hdf5-attrs`, or `nwb` for
   other NWB files). An NWB `trials` or `*stim*` interval table with parameter columns is read the
   same way.
2. **A parameter table** in `params`: one row per train (matched by time when it has an onset
   column, else in order), or one row per recording (`recording_id`, file name or subject column).
   Column names are recognised with their units: `current_uA`, `Amplitude (mA)`, `I`, `freq`,
   `Rate (kHz)`, `PW (us)`, `pulse_width_ms`, `duration_s`, `polarity`, `stim_shape`, `site`,
   `contact`, `condition`, `onset_s`, `stop`, ...; anything else is kept as an extra column in
   lowercase snake_case. `stim-map.yaml` names columns that are not recognised.
3. **Stimulation channels** (`stim` type in `channels.csv`), read in one of three ways (decided from
   the channel's name, else its content, or set in `stim-map.yaml`):
   - **trigger / TTL**: a two-level line. Threshold crossing with hysteresis (60 % / 40 % of the way
     from the resting to the active level, or `TRIGGER_THRESHOLD`); crossings closer than
     `PULSE_MERGE_S` are one pulse (lines that mark each pulse's start and end with a one-sample blip
     give the pulse width as the blips' spacing); pulses closer than `TRAIN_GAP_S` are one train.
     Rate = 1 / median inter-pulse interval; `n_pulses` counted; train duration = last onset − first
     onset + one interval. Three or more crossings at ≥ 1 kHz are a kilohertz burst: a `block` row.
     Two identical pulses in overlapping trial windows count once.
   - **stimulus monitor** (the current or voltage itself): the same on |x − baseline| (threshold
     10 × the noise, at least 2 % of the largest pulse), plus the amplitude (median pulse peak,
     converted to uA / mA / V), polarity (sign of the first phase; negative = cathodic) and
     waveform (one phase = monophasic, two phases of opposite sign = biphasic, with a gap =
     biphasic-interphase).
   - **envelope** (a stimulation setting recorded over time, as a closed-loop controller's output):
     epochs where |value| > `ENVELOPE_THRESHOLD` (default 10 % of the largest value); epochs of the
     same sign closer than `ENVELOPE_GAP_S` are joined (missing samples count as off); an epoch is
     split where its 1 s median steps by more than `ENVELOPE_STEP`; value per epoch = median. What a
     negative value means (e.g. kilohertz block) is said in `stim-map.yaml`.
4. **Imported markers** (physio-import's `events.tsv`, else the NWB `markers` table): with
   `MARKER_PATTERN=auto`, markers that name a stimulation (stim, VNS, train, pulse, block, ...) or
   carry values with units (`1.5mA 20Hz 250us`); "on" / "off" markers are paired into one epoch;
   "baseline" / "rest" markers become `baseline` rows. A regular expression with named groups
   (`amplitude_mA`, `frequency_hz`, `pulse_width_us`, `condition`, ...) reads other marker texts.
5. **File and folder names** of the source recording: with `NAME_PATTERN=auto`, values with units
   (`100uA_200us.abf`, a folder `10 Hz cervical VNS`); or a regular expression with named groups,
   e.g. `(?P<condition>gain\d+)` for dataset 655's folders.
6. **SPARC Data Structure metadata** (optional, `USE_SDS`): parameter columns of the
   `manifest` / `performances` / `samples` / `subjects` row of the recording, and values with units in
   their description columns.
7. **`stim-map.yaml` `defaults`**.

A table row, NWB row or marker matched to a detected train takes the train's timing (a channel
places a train to the sample; a log is usually rounded and a trial table gives the trial window),
and the row's parameters; rows that match no train are placed at their own time (and listed in
`qc.csv`). When nothing gives a time but a file name, SDS or table gives parameters, one
`condition` row spans the recording. Baseline epochs are written only where a source marks them;
otherwise stim-dose-response takes its own pre-stimulus window. A recording with no stimulation
source at all is listed as `skipped` in `provenance.json` (not a failure).

### stim-map.yaml

```yaml
version: 1
channels:                      # matched in order on the channel name, original name or description
  - match: CtrlAmp
    kind: envelope             # trigger | monitor | envelope | ignore
    quantity: amplitude        # envelope: amplitude | frequency_hz | pulse_width_us
    unit: mA                   # unit of the values (after scale)
    scale: 1.0
    threshold: 1.0             # overrides TRIGGER_/ENVELOPE_THRESHOLD for this channel
    site: right cervical vagus
    group: control             # channels with the same group describe one stimulator: one event
    positive: {trial_type: stimulation}
    negative: {trial_type: block, waveform: kilohertz}   # value becomes |value|
    set: {pulse_width_us: 50, waveform: biphasic}       # fixed parameters of this channel's events
tables:
  - file: "stim_log*.xlsx"     # glob on the file name or its path within params
    sheet: Sheet1
    columns: {amplitude: "I (mA)", onset: "t_on", frequency_hz: "Rate"}
    amplitude_unit: mA
    time_unit: s               # s | ms | min
    recording_column: File
    match: time                # time | order | recording
defaults: {site: cervical vagus}
```

The one used for dataset 655 is `validation/d655/stim-map.yaml`.

## Parameters

| Name | Default | What it does |
|---|---|---|
| `TRAIN_GAP_S` | 2 | Pulses further apart than this start a new train. |
| `PULSE_MERGE_S` | 0.002 | Crossings closer than this are one pulse (two phases, or start/end blips). |
| `TRIGGER_THRESHOLD` | auto | Trigger: 60 % of the way to the active level; monitor: 10 × noise, ≥ 2 % of the largest pulse. Or a number in the channel's unit. |
| `ENVELOPE_THRESHOLD` | auto | 10 % of the envelope's largest value, or a number. |
| `ENVELOPE_GAP_S` | 5 | Envelope epochs of the same sign closer than this are joined. |
| `ENVELOPE_STEP` | 0.25 | Split an envelope epoch where its 1 s median changes by more than this fraction. |
| `MIN_GAP_S` | 10 | Trains starting less than this after the previous one ends (or after the recording starts) are listed in `qc.csv`. 10 s matches the default baseline window of stim-dose-response. |
| `MATCH_TOLERANCE_S` | 2 | Table rows, NWB rows and markers with one time match the train starting within this. |
| `MARKER_PATTERN` | auto | `auto`, `none` or a regex with named groups (above). |
| `NAME_PATTERN` | auto | `auto`, `none` or a regex with named groups on the source path. |
| `USE_SDS` | true | Read SPARC Data Structure metadata in `params`. |
| `FIGURES` | true | One timeline figure per recording. |
| `THREADS` | 0 | BLAS threads (0 = from the container's CPU quota). |

No species preset is needed: detection works on the stimulation channels, which do not depend on
the species.

## Outputs (`events` port, under `stim-events/`)

- `events.tsv`: conventions.md's columns, in order: `recording_id, onset, duration` (s from the
  recording start), `trial_type` (`stimulation` | `block` | `baseline` | `condition`), `event_id`
  (`stim-0001`, `block-0001`, `base-0001`, `cond-0001`), `amplitude, amplitude_unit` (uA, mA, V; `n/a`
  when the channel or table gives no unit), `pulse_width_us` (one phase), `frequency_hz`,
  `train_duration_s, train_on_s, train_off_s, n_pulses, polarity, waveform, site, contact,
  condition, label, source` (every source that contributed, highest priority first, `|`-joined,
  e.g. `hdf5-attrs|trigger`), then any extra columns a table supplied. One row per train, not per
  pulse.
- `stim_summary.csv`: per recording: status, numbers of trains / blocks / baseline / condition rows,
  parameters varied, min/max amplitude, rate, pulse width and train duration, total stimulation time,
  sources.
- `qc.csv` (`recording_id, check, value, note`): `channel_kind` (how each stimulation channel was read
  and why), `table_vs_channel` (rate > 10 %, pulse width > 20 %, train duration > 0.5 s or 10 %,
  pulse count > 10 %, amplitude > 10 % apart: the table value is written), `unmatched_row`,
  `row_count`, `overlap`, `baseline_gap`, `envelope_negative`, `amplitude_unit_unknown`,
  `channel_note` (e.g. pulse rate varying within a train), `no_timing`.
- `features.csv` (`recording_id, channel, feature, value`): `stim_n_trains`, `stim_n_blocks`,
  `stim_total_s`, amplitude and rate ranges, numbers of amplitude and rate levels.
- `figures/<recording_id>_timeline.png`, `summary.json` (headline numbers, figure captions,
  warnings), `provenance.json`.

## Runtime

Measured in Docker (linux/amd64 under emulation on an M-series Mac, `--cpus 1 --memory 2g`,
read-only file system, arbitrary uid, as on Lambda) on physio-import's outputs of the demo data:

| Demo | Recordings | Wall time | Peak memory (cgroup) |
|---|---|---|---|
| d349 (pig, 432 s trial-by-trial, 30 kHz trigger line, 10.8 M samples) | 1 | 1.4 s (2.1 s with container start) | 171 MB |
| d655 (rat, 3 × 20–28 min, two 1 kHz envelopes) | 3 | 1.7 s | 162 MB |
| Fantasia (6 × 2 h, no stimulation) | 6 | 1.0 s | 180 MB |
| d406 (20 WAV ECG recordings, no stimulation) | 20 | 0.1 s | 55 MB |
| Autonomic Aging (3, no stimulation) | 3 | 0.1 s | 53 MB |

Chosen: 512 CPU units, 1024 MiB, 900 s timeout (fits Lambda). Time grows linearly with the number of
stimulation-channel samples; memory does not (slices of 1 M samples). Image: about 95 MB.

## Validation

**Discover 349** (pig, 20 trains on a 50–2500 µA × 2–30 Hz grid; `validation/validate_349.py`,
results in `validation/d349-results.txt`). The `stim_timing` line alone, before any table is used,
gives 1550 pulses (= 5 s × (2 + 10 + 20 + 30) Hz × 5 amplitudes), rates of exactly 2, 10, 20 and 30 Hz,
5.000 s trains and a 500 µs pulse width from the start/end blips, all equal to the HDF5 attributes:
no table vs channel disagreement. After the merge, all 20 rows carry the attributes' current, rate,
width, duration and polarity, and `n_pulses` = rate × duration. Onsets fall 11.6 to 14.2 ms (median
13.0 ms) before the time in each trial group's name (measured on the source's own time stamps;
`validate_349.py` reports 12.3 to 15.0 ms because recordings.csv `start_time` is rounded to the
millisecond). QC flags
7 trains that start 4.2–10.0 s after the previous one ends: their 10 s pre-stimulus windows include
the previous train (the trials' recorded windows overlap in this dataset).

**Discover 655** (rat closed-loop control; `validation/validate_655.py`, results in
`validation/d655-results.txt`), with `validation/d655/stim-map.yaml` and
`NAME_PATTERN=(?P<condition>gain\d+)`: in all three recordings the perturbation electrode's
envelope gives one 593.9 s epoch (the protocol's 10-minute 1 → 30 → 1 Hz ramp, from its first to its
last pulse) starting 123.3, 124.1 and 124.8 s after the controller's first output (protocol: 2
minutes); the control electrode gives stimulation (positive) then kilohertz block (negative) epochs,
and in two of three recordings stays on 121.8 and 121.7 s after the ramp (protocol: 2 minutes; in
the gain-2 recording it stops 2.8 s after the ramp). The gain-5 and gain-10 recordings also show a
44 s and 61 s perturbation-electrode epoch before the controller starts (the protocol's initial
maximal stimulation). Amplitudes are written without a unit (the exported envelopes have none).

**Synthetic fixture** (`tests/make_fixture.py`, `make test`): a TTL line with a parameter log
(timing from the line, amplitudes from the log, one planted 10 vs 20 Hz disagreement found), a
trial-by-trial recording with HDF5-style per-trial parameters and start/end blips (onsets within
0.2 ms), envelopes with stimulation and block, on/off markers with values in the text, a biphasic
current monitor (800 µA, 200 µs, 30 Hz, cathodic first, interphase gap: all recovered), and a
recording whose parameters are only in its file name.

## Limitations

- An event's parameters are only as good as their source: a trigger line gives timing, rate and
  pulse count but no amplitude; an envelope gives a setting, not the current delivered. Say units in
  `stim-map.yaml` when the channel has none, or amplitudes cannot be compared across recordings.
- Monitor-derived amplitude, width and polarity assume the monitor shows the stimulus without large
  artefact or filtering; polarity assumes negative = cathodic at the stimulating contact.
- Envelope epochs are where a setting is non-zero; how the stimulator turned that setting into pulses
  (rate, width) is not in the envelope.
- Detection does not depend on species or anaesthesia, but the `MIN_GAP_S` check is only as right as
  the baseline window you will use.
- Tested on the two demo stimulation datasets above and synthetic data; the readers of other
  acquisition systems' trigger conventions (active-low lines, burst-coded markers) are covered by the
  generic rules but not by a public dataset.
- Research output, not a clinical finding.

## Licences

This app: Apache-2.0. Bundled packages: BSD-3-Clause, MIT, MIT-CMU, Apache-2.0 and the Matplotlib
licence (`THIRD_PARTY.md`). Demo data: CC BY 4.0 (Discover 349 and 655).

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author
lists in CITATION.cff.

- Rübel O, Tritt A, Ly R, Dichter BK, Ghosh S, Niu L, Baker P, Soltesz I, Ng L, Svoboda K, Frank L,
  Bouchard KE (2022). The Neurodata Without Borders ecosystem for neurophysiological data science.
  *eLife* 11:e78362. doi:10.7554/eLife.78362, PMID 36193886.
- Gorgolewski KJ et al. (2016). The brain imaging data structure, a format for organizing and
  describing outputs of neuroimaging experiments. *Scientific Data* 3:160044.
  doi:10.1038/sdata.2016.44, PMID 27326542.
- Bandrowski A et al. (2021). SPARC Data Structure: Rationale and Design of a FAIR Standard for
  Biomedical Research Data. bioRxiv. doi:10.1101/2021.02.10.430563 (when SDS metadata is used).
- Harris CR et al. (2020). Array programming with NumPy. *Nature* 585:357-362.
  doi:10.1038/s41586-020-2649-2, PMID 32939066.
- Hunter JD (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*
  9:90-95. doi:10.1109/MCSE.2007.55.
- h5py (Andrew Collette and contributors), https://github.com/h5py/h5py; openpyxl, PyYAML: see
  CITATION.cff.

Demo data (please cite when you use them):

- Armitage O, Sarkans E, Wernisch L, Fortier-Poisson P, Berthon A, Tessier-Larivière O (2023).
  Influence of vagus nerve stimulation on cardiac activity in pigs. Pennsieve Discover (SPARC
  Portal). doi:10.26275/dm5l-tspb.
- Bender SA, Green DB, Thakkar VS, Zimmerman HL, Elazab M, Kilgore KL, Vrabec TL (2026).
  Single-electrode bidirectional control of heart rate via vagus nerve modulation in rat model.
  Pennsieve Discover (SPARC Portal). doi:10.26275/5cwo-mkpz.

## Acknowledgements

NWB is developed by the Neurodata Without Borders project; h5py, NumPy, openpyxl, PyYAML and
Matplotlib by their communities. The demo datasets are shared by their authors on Pennsieve Discover
under CC BY 4.0.
