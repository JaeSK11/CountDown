# CountDown training image: the exact package set of countdown/.venv (torch 2.14 + CUDA 13.0,
# needed for the RTX 5090's sm_120).  Code, data, caches and runs are NOT baked in -- the
# whole checkout is bind-mounted at its host path (compose.yaml), so absolute paths in the
# configs and the data/ symlinks resolve exactly as on the host and edits need no rebuild.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# procps: pgrep/ps for the queue runner; libpcap0.8: scapy's optional capture backend;
# libgomp1: LightGBM's OpenMP runtime (torch vendors its own copy, which hid this in tests).
RUN apt-get update \
 && apt-get install -y --no-install-recommends procps libpcap0.8 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r /tmp/requirements.lock \
 && python -c "import torch, torch_geometric, lightgbm; print(torch.__version__, torch.version.cuda)"

# Live capture (`countdown live`) opens a raw AF_PACKET socket, which needs CAP_NET_RAW. The
# container runs as uid 1000 (so outputs stay owned by the user), and a non-root process does
# not inherit container capabilities -- so a *copy* of the interpreter carries the one file
# capability, and only `countdown live` runs on it. Nothing else in the image gets raw sockets.
RUN apt-get update && apt-get install -y --no-install-recommends libcap2-bin \
 && cp "$(readlink -f /usr/local/bin/python3)" /usr/local/bin/python-netraw \
 && setcap cap_net_raw+ep /usr/local/bin/python-netraw \
 && apt-get purge -y libcap2-bin && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# `countdown` on PATH, as `pip install -e` gives the host venv (the project is not installed here).
RUN printf '#!/bin/sh\nif [ "$1" = live ]; then exec python-netraw -m countdown "$@"; fi\nexec python -m countdown "$@"\n' \
      > /usr/local/bin/countdown \
 && chmod +x /usr/local/bin/countdown

# The project itself comes from the bind mount; PYTHONPATH stands in for `pip install -e`.
ENV PYTHONPATH=/home/counterbitio/CountDown/countdown \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/mpl
WORKDIR /home/counterbitio/CountDown/countdown
