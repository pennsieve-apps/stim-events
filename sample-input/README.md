# Sample input

stim-events reads physio-import's `physio` output (port `physio`) and, optionally, parameter tables,
a `stim-map.yaml` and SPARC Data Structure metadata (port `params`).

## Synthetic (no download)

    python -m tests.make_fixture data/fixture   # data/fixture/physio and data/fixture/params

## Real recordings used to test this app

Run physio-import on these first; its `physio` folder is this app's `physio` input.

- **Pennsieve Discover 349** (doi:10.26275/dm5l-tspb, CC BY 4.0): pig vagus nerve stimulation grid,
  `files/primary/sub-2/subject-2.hdf5` (496 MB; per-trial parameters in the HDF5 group attributes,
  the pulse line `neural/stim_timing` at 30 kHz). No `params` needed.
- **Pennsieve Discover 655** (doi:10.26275/5cwo-mkpz, CC BY 4.0): rat closed-loop heart-rate control,
  the CSV exports `files/primary/sub-Bimodal01/perf-*/*.smrx.csv` (envelopes `4 StimAmp`, `5 CtrlAmp`).
  Put `validation/d655/stim-map.yaml` in `params` and set `NAME_PATTERN=(?P<condition>gain\d+)`.

In the workflow repo, `scripts/fetch-sample-data.sh` stages both under `data/`; physio-import's and
this app's outputs on them are under `data/work/<demo>/`.
