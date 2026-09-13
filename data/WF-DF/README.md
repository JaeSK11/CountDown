# DF — Deep Fingerprinting Dataset (Tor WF, single-tab)

**Status:** ⏳ Downloading (Zenodo, direct). Files land here as `*.npz.zip`.
**Primary source (convenient .npz):** WFlib Zenodo record — https://zenodo.org/records/13732130
**Original release:** https://github.com/deep-fingerprinting/df (pickle files on Google Drive:
https://drive.google.com/drive/folders/1kxjqlaWWJbMW56E3kbXlttqcKJFkeCy6)
**Paper:** Sirinam, Imani, Juarez, Wright, "Deep Fingerprinting: Undermining Website
Fingerprinting Defenses with Deep Learning," ACM CCS 2018. DOI: 10.1145/3243734.3243768

## What it is
The standard Tor **website-fingerprinting** benchmark: closed-world **95 monitored websites**
(DF paper). Traces are directional packet sequences (±1). Used single-tab, closed-world.

## Files pulled here (from Zenodo, zipped .npz)
| File | Size | Content |
| --- | --- | --- |
| `CW.npz.zip` | 1.3 GB | Closed-world, **undefended** (95 sites, 105,730 traces) |
| `WTF-PAD.npz.zip` | 2.2 GB | Same traffic under **WTF-PAD** defense |
| `Front.npz.zip` | 1.3 GB | Under **FRONT** defense |
| `TrafficSliver.npz.zip` | 0.74 GB | Under **TrafficSliver** defense |

`.npz` = NumPy archive (load with `numpy.load`). Unzip each `.npz.zip` first.

## How the paper uses it
"Hybrid CNN-SSM Model for Robust Multi-Tab Website Fingerprinting over Tor" (IJCNN/WCCI 2026):
single-tab closed-world baseline on DF (undefended) and DF under simulated defences
(WTF-PAD, FRONT, TrafficSliver, RegularTor). **Note:** RegularTor is *not* on Zenodo — the
paper simulates it on DF; regenerate from the DF traces if needed.

## After downloading
`unzip '*.npz.zip'` here; record the array keys/shapes below.
