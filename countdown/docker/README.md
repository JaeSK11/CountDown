# Running CountDown in Docker

Long training jobs run in a container so that nothing ties them to a terminal or a Claude
session: on 2026-10-08 a GPU queue started from a session was killed when that session ended,
taking ~1.5 h of GIN folds with it.

| File | What it is |
| --- | --- |
| `Dockerfile` | `python:3.12-slim` + the exact package set of `countdown/.venv` (torch 2.14 + CUDA 13.0, which the RTX 5090's `sm_120` needs). Code and data are **not** baked in. |
| `requirements.lock` | `pip freeze` of `.venv`, minus the editable project and the unused CUDA-12 wheels the torch upgrade left behind. |
| `compose.yaml` | `gpu-queue` (the queue daemon) and `job` (one-off commands). Both bind-mount `/home/counterbitio/CountDown` at the same path, run as uid 1000, see all GPUs, and get a 32 GB `/dev/shm` for DataLoader workers. |
| `gpu-queue-runner.sh` | The queue: one job at a time from `runs/gpu-queue/pending/`, lowest number first. |
| `census` service | the weekly PQC census daemon (below); no GPU. |
| (in the image) `python-netraw` | interpreter copy with `cap_net_raw` for `countdown live` only (below). |

## Use

```bash
cd countdown
docker compose -f docker/compose.yaml build                     # after changing requirements.lock / Dockerfile
docker compose -f docker/compose.yaml up -d gpu-queue           # start (restarts itself after a reboot)
docker compose -f docker/compose.yaml stop gpu-queue            # stop; stays stopped until `up -d`
docker compose -f docker/compose.yaml run --rm job python -m pytest -q     # anything one-off
docker logs countdown-gpu-queue                                 # the runner's own output
tail runs/gpu-queue/queue.log                                   # START / EXIT <rc> per job
```

**Adding work:** drop `NN-name.sh` into `runs/gpu-queue/pending/`. Call Python as
`${PY:-.venv/bin/python}` — the container sets `PY=python`, because the host venv's
interpreter does not exist inside the image; the host fallback still uses `.venv`.

## Guarantees and limits

- **Survives** session ends, terminal closes, SSH drops (the container belongs to Docker) and
  reboots (`restart: unless-stopped`; Docker starts at boot).
- **Interrupted jobs resume.** A job sits in `running/` while it runs; on restart the runner
  moves it back to `pending/`. Phase-5 ensemble jobs cache every finished fold, so the re-run
  redoes at most one fold.
- **One GPU job at a time**, on purpose: a GIN fold alone fills ~29.6 of 32.6 GB, and this
  5090 has dropped off the PCIe bus under load (`phases/PHASE-4-HANDOFF.md` §0).
- **No GPU telemetry inside the container.** The `~/gpu-watch` PCIe/power logger needs host
  `sudo`; run it on the host. The container's queue log says `Xid: n/a`; check the host with
  `journalctl -k | grep -i 'nvrm: xid'` (plain `xid` also matches the r8169 NIC's boot line).

## Live capture

`countdown live <iface>` reads a raw `AF_PACKET` socket, which needs `CAP_NET_RAW`. The container
runs as uid 1000, and a non-root process does not inherit container capabilities, so the image
carries a **copy** of the interpreter, `python-netraw`, with the single file capability
`cap_net_raw+ep`. The `countdown` wrapper runs only `countdown live` on it; nothing else in the
image (and nothing on the host) gets raw sockets.

```bash
# Its own traffic only (default bridge network) -- what the 2026-10-09 demo used:
docker compose -f docker/compose.yaml run --rm job countdown live eth0 --port 443 --duration 30
# The host's interfaces (deployment) -- sees ALL host traffic; only where you may monitor:
docker compose -f docker/compose.yaml run --rm live countdown live eth0 --port 443
```

`--mode fast` (the default) uses only each expert's GBDT; `--mode full` adds E2's graph member,
fed by a streaming byte collector. Filtering is by port in Python; kernel BPF is not supported
(it needs libpcap, which the image does not carry).

## Weekly PQC census

```bash
docker compose -f docker/compose.yaml up -d census       # start (restarts itself after a reboot)
docker compose -f docker/compose.yaml stop census        # stop
docker compose -f docker/compose.yaml run --rm census python scripts/census_scheduler.py --status
tail runs/census/scheduler.log                           # START / DONE / FAIL / TREND per run
```

`countdown-census` runs `scripts/census_scheduler.py`: every 7 days at 03:00 UTC
(`CENSUS_EVERY_DAYS`, `CENSUS_HOUR_UTC` in `compose.yaml`) it runs the TCP + QUIC census on the
fixed Tranco list into `results/census/runs/<date>/` and rebuilds `results/census/trend.*`. It
uses the default bridge network (its capture sees only its own probes) and no GPU, so it runs
alongside the GPU queue. A run counts only once it finishes (`<date>.partial` until then); a
failed run is retried after 6 h. **It commits nothing**: review and commit the new run yourself.

## Rebuilding the lock

```bash
.venv/bin/python -m pip freeze --exclude-editable | grep -v -- '-cu12==' > docker/requirements.lock
```

then re-add the two header comment lines and rebuild. After any image change, import every
native dependency **in a fresh interpreter** (`python -c "import lightgbm"`, …): torch ships
its own `libgomp`, which hid LightGBM's missing OpenMP runtime from the whole test suite once.
