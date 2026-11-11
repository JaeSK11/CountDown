# CountDown

Research code for **CountDown**: detect the cryptography in use on a network flow, decide
whether it is quantum-resistant, and — for flows whose crypto is not observable — classify
the encrypted traffic with an ensemble of ML models.

```
PIPELINE:   [1 Ingest] → [2 Crypto detect] → [3 PQC verdict] → [4 ML ensemble]
```

Stages 2–3 are deterministic parsing, no ML: quantum-resistance is read from a directly
observable TLS/QUIC handshake. Tunnelled traffic (VPN/Tor) gets the verdict *"crypto not
observable"* and routes straight to stage 4.

The full write-up lives in [`countdown/README.md`](countdown/README.md); the roadmap is in
[`countdown/PLAN.md`](countdown/PLAN.md) with per-phase specs under `countdown/phases/`.

## Layout

| Path | Contents |
| --- | --- |
| `countdown/` | The package, scripts, configs and tests — everything in this repo |
| `data/` | **Not included.** One folder per dataset, each with a `README.md` giving the source, licence and download steps |
| `reference/` | **Not included.** Index only — see `reference/README.md` |

## Getting started

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e countdown
pytest countdown/tests -q
```

The test suite runs on small synthetic fixtures checked into
`countdown/tests/golden/`, so it passes without downloading anything.

## Data

No datasets are redistributed here. They belong to their original authors, several are
gated behind registration or an email request, and together they run to roughly 72 GB.

**[`data/README.md`](data/README.md) is the index** — it lists every dataset, its task,
how to obtain it and which paper uses it. Each `data/<dataset>/README.md` then carries the
direct source, licence terms and step-by-step download instructions. Recreate the folder
layout as documented and the loaders will find the files.

Likewise `reference/` is not included: it holds copyrighted papers, upstream repositories
under their own licences (ET-BERT, encrypted-traffic-classifier) and ~683 MB of pretrained
weights. [`reference/README.md`](reference/README.md) records what each item is and where
it came from, so you can fetch them yourself.

## Licence

Copyright (C) 2026 JaeSK11.

The code in this repository is free software: you can redistribute it and/or modify it under
the terms of the [GNU General Public License, version 3](LICENSE) (GPL-3.0-only). It is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the licence for
details.

That covers this code only — datasets, papers and pretrained models referenced above remain
under the terms set by their respective authors. Models trained on those datasets (the
`models-v1` release) may carry the datasets' terms as well.
