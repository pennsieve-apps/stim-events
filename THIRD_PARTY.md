# Third-party components

The Docker image bundles these Python packages (exact versions in `requirements.txt`), installed
from PyPI wheels on `python:3.12.16-slim-bookworm` (Debian bookworm with CPython 3.12, PSF licence).
No data, models or other people's analysis code are bundled. The stimulation detection, the merging
of sources and the QC checks are this app's own code (Apache-2.0).

| Component | Version | Authors / copyright holder | Licence | URL |
|---|---|---|---|---|
| h5py | 3.16.0 | Andrew Collette and the h5py contributors | BSD-3-Clause | https://github.com/h5py/h5py |
| NumPy | 2.5.4 | NumPy Developers (Travis E. Oliphant et al.) | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| openpyxl | 3.1.5 | Charlie Clark and the openpyxl contributors | MIT | https://foss.heptapod.net/openpyxl/openpyxl |
| et-xmlfile | 2.0.0 | Charlie Clark and the et-xmlfile contributors | MIT | https://foss.heptapod.net/openpyxl/et_xmlfile |
| PyYAML | 6.0.3 | Kirill Simonov and the PyYAML contributors | MIT | https://github.com/yaml/pyyaml |
| Matplotlib | 3.11.2 | John D. Hunter, Michael Droettboom and the Matplotlib Development Team | Matplotlib License (PSF-based) | https://matplotlib.org |
| contourpy | 1.4.0 | Ian Thomas and the ContourPy developers | BSD-3-Clause | https://github.com/contourpy/contourpy |
| cycler | 0.12.1 | Matplotlib project (Thomas A Caswell) | BSD-3-Clause | https://github.com/matplotlib/cycler |
| fonttools | 4.68.0 | Just van Rossum, Behdad Esfahbod and fontTools contributors | MIT | https://github.com/fonttools/fonttools |
| kiwisolver | 1.5.1 | Nucleic Development Team | BSD-3-Clause | https://github.com/nucleic/kiwi |
| packaging | 26.3 | Donald Stufft and contributors | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| Pillow | 13.0.0 | Jeffrey A. Clark and contributors | MIT-CMU | https://github.com/python-pillow/Pillow |
| pyparsing | 3.3.3 | Paul McGuire | MIT | https://github.com/pyparsing/pyparsing |
| python-dateutil | 4.0.0 | Gustavo Niemeyer, Paul Ganssle and dateutil contributors | Apache-2.0 AND BSD-3-Clause | https://github.com/dateutil/dateutil |
| tzdata | 2026.6 | Python Software Foundation | Apache-2.0 | https://github.com/python/tzdata |

Test-only (not in the image): pytest (MIT); pynwb (BSD-3-Clause, Neurodata Without Borders) writes
the synthetic fixture's NWB files.

Formats and conventions followed (not code): NWB 2.x (Rübel et al. 2022) for the input files, BIDS
`events.tsv` (Gorgolewski et al. 2016) for the output table, the SPARC Data Structure (Bandrowski
et al. 2021) as one optional metadata source.
