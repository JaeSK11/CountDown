# ARES — Multi-tab Website Fingerprinting Dataset (Tor)

**Status:** ⏳ Downloading (Zenodo, direct). Files land here as `closed_Ntab.npz.zip` /
`open_Ntab.npz.zip`.
**Source:** WFlib Zenodo record — https://zenodo.org/records/13732130
**Repos:** https://github.com/Xinhao-Deng/Multitab-WF-Datasets ·
https://github.com/Xinhao-Deng/Website-Fingerprinting-Library (WFlib)
**Paper:** Deng et al., "Robust Multi-tab Website Fingerprinting Attacks in the Wild," IEEE S&P
2023 (extended: arXiv:2501.12622). Tsinghua University.

## What it is
Real-world **multi-tab** Tor traffic: traces with **2–5 concurrently open tabs**, from Alexa Top
sites. Both **closed-world** (monitored = Alexa top 100) and **open-world** (k−1 monitored + 1
non-monitored from Alexa top 20,000) splits. Improved/denoised version (ResNet screenshot
filtering) per the extended paper.

## Files pulled here (from Zenodo, zipped .npz)
| File | Size | | File | Size |
| --- | --- | --- | --- | --- |
| `closed_2tab.npz.zip` | 1.1 GB | | `open_2tab.npz.zip` | 1.3 GB |
| `closed_3tab.npz.zip` | 1.4 GB | | `open_3tab.npz.zip` | 1.6 GB |
| `closed_4tab.npz.zip` | 1.6 GB | | `open_4tab.npz.zip` | 1.7 GB |
| `closed_5tab.npz.zip` | 1.6 GB | | `open_5tab.npz.zip` | 1.8 GB |

`.npz` = NumPy archive (`numpy.load`). Unzip each `.npz.zip` first.

## How the paper uses it
"Hybrid CNN-SSM Model for Robust Multi-Tab Website Fingerprinting over Tor" (IJCNN/WCCI 2026):
**the primary multi-tab evaluation** — unknown number of open tabs, closed + open world.

## After downloading
`unzip '*.npz.zip'` here; record array keys (traces/labels), tab-count layout, and shapes below.
