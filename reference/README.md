# Reference Index

Papers and code collected for the encrypted-traffic-classification / website-fingerprinting
modeling work. Each paper in `papers/` has a `.pdf` plus a `.md` sidecar note (why it's here,
method, datasets → matching `../data/` folders). Datasets index: `../data/README.md`.
_Last updated: 2026-08-23._

## Papers (`papers/`)

| Paper | Venue / Year | Focus | Datasets used | → data folders |
| --- | --- | --- | --- | --- |
| **Characterization of Encrypted and VPN Traffic using Time-related Features** (Draper-Gil) | ICISSP 2016 | Flow time-based features; C4.5 + k-NN; **origin of ISCXVPN2016** | ISCXVPN2016 (own capture, 28 GB) | `ISCXVPN2016` |
| **CNN-Based Encrypted Network Traffic Classifier** (Okonkwo) | AISC 2022 | Flows→images CNN; https/VPN/Tor | ISCXVPN2016, ISCXTor2016 | `ISCXVPN2016`, `ISCXTor2016` |
| **Deep Learning for Encrypted Traffic Classification and Unknown Data Detection** | Sensors 2022 | In-app activity + open-set detection | Self-collected (8 apps, 92 activities) | `MobileAppActivity-Sensors2022` |
| **Lightweight Graph Representation encrypted-traffic classifier** | Sci. Reports 2025 | Graph-based encrypted-traffic classifier | ISCX-2012, WWT, ISCX-Tor | `ISCXIDS2012`, `WWT`, `ISCXTor2016` |
| **Generalized Encrypted Traffic Classification Using Inter-Flow Signals** | arXiv 2025 (2508.21558) | Cross-task generalization (8 datasets) | ISCX-VPN/nonVPN, MAppGraph, PostQuantumTLS, Cross Platform, CICAndMal2017, IoT-Sentinel, CSTNET-TLS1.3 | `ISCXVPN2016`, `MAppGraph`, `PostQuantumTLS`, `CrossPlatform`, `CICAndMal2017`, `IoT-Sentinel`, `CSTNET-TLS1.3` |
| **Hybrid CNN-SSM Model for Multi-Tab Website Fingerprinting over Tor** (Okonkwo) | IJCNN/WCCI 2026 | Multi-tab Tor WF (CNN + state-space/Mamba) | DF, Walkie-Talkie, Wang, ARES | `WF-DF`, `WF-WalkieTalkie`, `WF-Wang`, `WF-ARES` |
| **Structure-Aware & Explainable WF using GNNs** (Okonkwo) | Information Sciences 2026 | Single-tab Tor WF via graphs + explainability | DF, Walkie-Talkie, Wang (+ defended DF) | `WF-DF`, `WF-WalkieTalkie`, `WF-Wang` |
| **ET-BERT: Contextualized Datagram Representation w/ Pre-training Transformers** (Lin et al.) | WWW 2022 | Raw-byte BERT (BURST tokens); pretrain → fine-tune | ISCX-VPN, ISCX-Tor, USTC-TFC, Cross-Platform, CSTNET-TLS1.3 (theirs) | `ISCXVPN2016`, `ISCXTor2016`, `CrossPlatform`, `CSTNET-TLS1.3` |

Four of the eight are from the same **Griffith University / Okonkwo** group (the two WF papers,
the CNN classifier, and — collaborators — the Hybrid CNN-SSM).

## Code (`encrypted-traffic-classifier/`)
Cloned repo (RADCOM competition, distributed CNN/Celery/FastAPI system). Uses a **private RADCOM
dataset** (not public). Reference for system design + Random-Forest feature engineering
(TLS first-packet-size fingerprinting, silence windows, inter-arrival times).

## Themes
- **Encrypted-traffic / app classification:** Draper-Gil (flow-stats baseline), CNN-Okonkwo, Sensors 2022, Lightweight-Graph, Generalized.
- **Raw-byte / pre-trained models:** ET-BERT (bytes member baseline; CSTNET-TLS1.3 origin).
- **Tor website fingerprinting:** Hybrid CNN-SSM, Structure-Aware-GNN.
