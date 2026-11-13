FROM --platform=linux/amd64 python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# Pinned Python dependencies, binary wheels only, no dependency resolution at build time
# (requirements.txt lists every package). Test folders are removed. No data or models: the app only
# reads its inputs.
COPY requirements.txt /processor/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: --root-user-action=ignore -r /processor/requirements.txt \
    && find /usr/local/lib/python3.12 -depth -type d -name __pycache__ -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/numpy/*/tests /usr/local/lib/python3.12/site-packages/matplotlib/tests \
              /usr/local/lib/python3.12/site-packages/h5py/tests /usr/local/lib/python3.12/site-packages/PIL/tests \
    && python3 -c "import h5py, numpy, openpyxl, yaml, matplotlib; print('h5py', h5py.__version__, 'numpy', numpy.__version__)" \
    && mkdir -p /opt/stim-events/matplotlib \
    && MPLCONFIGDIR=/opt/stim-events/matplotlib python3 -c "import matplotlib.font_manager" \
    && ls /opt/stim-events/matplotlib/fontlist-*.json \
    && chmod -R a+rX /opt/stim-events

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

# The Matplotlib font cache is built above into a read-only path; at start the app copies it to
# $TMPDIR/mpl (MPLCONFIGDIR) because HOME and /tmp are replaced at run time on Lambda.
ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STIM_EVENTS_MPL_CACHE=/opt/stim-events/matplotlib \
    MPLBACKEND=Agg

# uid 1000 on ECS; any uid works (nothing in the image is written at run time).
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app
USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
