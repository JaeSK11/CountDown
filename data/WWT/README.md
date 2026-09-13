# WWT Dataset (WeChat, WhatsApp, Telegram) — NOT PUBLICLY AVAILABLE

> ⚠️ **This folder is a placeholder.** As of 2026-08-14 the WWT dataset has **no public
> download**. The data files are not stored here. See "How to obtain" below.

## What it is
Self-collected, fine-grained **user-behavior** encrypted-traffic dataset for three messaging
apps. Introduced by the **TFE-GNN** paper (WWW'23) and reused by the Nature 2025 paper.

| App | User-behavior categories |
| --- | --- |
| WeChat | 12 |
| WhatsApp | 9 |
| Telegram | 6 |

Distinguishing feature: unlike the public ISCX datasets, WWT additionally records the
**start/end timestamps** of each user-behavior sample, enabling traffic segmentation.

## Origin / papers
- **TFE-GNN**: "A Temporal Fusion Encoder Using Graph Neural Networks for Fine-grained
  Encrypted Traffic Classification," WWW 2023.
  - Paper: https://dl.acm.org/doi/abs/10.1145/3543507.3583227 · arXiv: https://arxiv.org/abs/2307.16713
  - Repo: https://github.com/ViktorAxelsen/TFE-GNN
  - Follow-ups: https://github.com/ViktorAxelsen/CLE-TFE · https://github.com/ViktorAxelsen/MH-Net
- Reused in: "Encrypted traffic classification encoder based on lightweight graph
  representation," *Scientific Reports* 2025 (attributed there to Wuhan University of
  Technology). See `reference/papers/Lightweight-graph-representation-...`.

## How to obtain
No public mirror exists (checked TFE-GNN / CLE-TFE / MH-Net repos — they only ship the
public ISCX datasets + `CATE/` label lists, no WWT data or download link).

To acquire it you must **request access from the authors** (TFE-GNN group). Options:
1. Open an issue on the TFE-GNN GitHub repo asking about WWT dataset availability.
2. Email the corresponding author listed on the TFE-GNN / arXiv paper.

Once obtained, drop the files here (`data/WWT/`) and update this README with the actual
format (pcap / npy / csv), sample counts, and train/test split.

## Alternative
If access can't be secured, the **public ISCX datasets** (see `data/ISCXVPN2016/`,
plus ISCX-Tor at unb.ca/cic/datasets/tor.html) are the drop-in substitutes those same
papers benchmark on.
