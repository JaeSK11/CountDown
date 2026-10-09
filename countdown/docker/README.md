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

## Rebuilding the lock

```bash
.venv/bin/python -m pip freeze --exclude-editable | grep -v -- '-cu12==' > docker/requirements.lock
```

then re-add the two header comment lines and rebuild. After any image change, import every
native dependency **in a fresh interpreter** (`python -c "import lightgbm"`, …): torch ships
its own `libgomp`, which hid LightGBM's missing OpenMP runtime from the whole test suite once.
