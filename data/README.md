# Datasets Index

Overview of every dataset folder, its task, access status, local state, and which paper(s) in
`../reference/papers/` use it. Each folder has its own `README.md` with full detail.
_Last updated: 2026-08-14._

Status legend: ✅ present locally · ⏳ in progress / blocked · 📝 documented, manual pull needed ·
🔒 gated (registration/email) · ❌ unavailable.

## Encrypted-traffic / app classification

| Folder | Dataset | Task | Access | Local | Used by |
| --- | --- | --- | --- | --- | --- |
| `ISCXVPN2016/` | ISCX VPN-nonVPN | VPN/non-VPN traffic classification | CIC form | ✅ ~26 GB | CNN-Okonkwo; Generalized (as ISCX-VPN/nonVPN) |
| `ISCXTor2016/` | ISCX Tor-nonTor | Tor/non-Tor characterization | CIC form | ✅ ~7.6 GB | CNN-Okonkwo; Lightweight-Graph |
| `ISCXIDS2012/` | ISCX IDS 2012 | Intrusion detection (7-day) | CIC form | 📝 placeholder | Lightweight-Graph |
| `CICAndMal2017/` | CIC Android Malware 2017 | Android malware detection | 🔒 CIC form | 📝 placeholder | Generalized |
| `MAppGraph/` | MAppGraph | Mobile-app fingerprinting | 🔒 email (58.5 GB) | 📝 placeholder | Generalized |
| `PostQuantumTLS/` | PostQuantumTLS (Zenodo 7950522) | Post-quantum TLS traffic | open | ✅ 4.2 GB (92 pcaps) | Generalized |
| `CrossPlatform/` | Cross Platform (ReCon) | Cross-platform app classification | ❌ withdrawn | — | Generalized |
| `IoT-Sentinel/` | IoT-Sentinel (Aalto) | IoT device fingerprinting | 📝 browser (Cloudflare) | 📝 placeholder | Generalized |
| `CSTNET-TLS1.3/` | CSTNET-TLS 1.3 (ET-BERT) | TLS 1.3 website fingerprinting | Google Drive | ✅ ~1.9 GB | Generalized |
| `WWT/` | WeChat/WhatsApp/Telegram | In-app behavior classification | ❌ private (request authors) | — | Lightweight-Graph |
| `MobileAppActivity-Sensors2022/` | Self-collected in-app activity | In-app activity + unknown detection | open (Dropbox) | ✅ 37 MB (368 CSVs) | Deep-Learning-Unknown |

## Tor website fingerprinting (WF)

| Folder | Dataset | Task | Access | Local | Used by |
| --- | --- | --- | --- | --- | --- |
| `WF-DF/` | DF (Deep Fingerprinting) | Single-tab WF (95 sites) + defended | Zenodo 13732130 | ⏳ blocked* | Hybrid-CNN-SSM; Structure-Aware-GNN |
| `WF-WalkieTalkie/` | Walkie-Talkie | WF defended | Zenodo 13732130 | ⏳ blocked* | Hybrid-CNN-SSM; Structure-Aware-GNN |
| `WF-Wang/` | Wang (kNN 100×100) | Single-tab WF | open (Tao Wang) | ✅ 74 MB | Hybrid-CNN-SSM; Structure-Aware-GNN |
| `WF-ARES/` | ARES | Multi-tab WF (2–5 tabs, closed+open) | Zenodo 13732130 | ⏳ blocked* | Hybrid-CNN-SSM |

\* **Zenodo IP block:** an initial burst of requests tripped Zenodo's abuse protection
("unusual traffic"). A paced retry is running but the block is persistent. Fastest fix: download
the needed `*.npz.zip` from https://zenodo.org/records/13732130 in a browser, or retry later.
Files needed: `CW/WTF-PAD/Front/TrafficSliver` (→WF-DF), `Walkie-Talkie` (→WF-WalkieTalkie),
`closed_2-5tab` + `open_2-5tab` (→WF-ARES).

## Notes
- ISCXVPN2016 covers both "ISCX-VPN" and "ISCX-nonVPN" referenced separately in some papers.
- DF/Walkie-Talkie/Wang are shared across the two Griffith WF papers; ARES is multi-tab only.
- Some defended variants (RegularTor, Tamaraw, CS-BuFLO) are *generated* by papers from DF/Wang,
  not downloaded.
