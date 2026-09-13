# Walkie-Talkie Dataset (Tor WF defense)

**Status:** ⏳ Downloading (Zenodo, direct) → `Walkie-Talkie.npz.zip`.
**Source (convenient .npz):** WFlib Zenodo record — https://zenodo.org/records/13732130
**Original traces:** Tao Wang's WF page — https://www.cs.sfu.ca/~taowang/wf/data/
(`walkiebatch.zip`, `walkiebatch-defended.zip`, `walkiebatch-burst.zip`).
**Paper:** Wang & Goldberg, "Walkie-Talkie: An Efficient Defense Against Passive Website
Fingerprinting Attacks," USENIX Security 2017.

## What it is
Tor traffic under the **Walkie-Talkie (WT)** defense (half-duplex + burst molding). The Zenodo
`Walkie-Talkie.npz` has **100 websites, 90,000 traces**.

| File | Size |
| --- | --- |
| `Walkie-Talkie.npz.zip` | 1.8 GB |

## How the paper uses it
Defended single-tab evaluation (the released WT-protected traces).

## After downloading
`unzip Walkie-Talkie.npz.zip`; load `.npz` with `numpy.load`. Record keys/shapes below.
