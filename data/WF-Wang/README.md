# Wang Dataset (Tor WF, kNN closed-world)

**Status:** ⏳ Downloading (direct) → `knndata.zip` (+ `knnsitelist.txt`).
**Source:** Tao Wang's WF page — https://www.cs.sfu.ca/~taowang/wf/data/knndata.zip (77.5 MB)
Site list: https://www.cs.sfu.ca/~taowang/wf/data/knnsitelist.txt
**Paper:** Wang, Cai, Nithyanand, Johnson, Goldberg, "Effective Attacks and Provable Defenses
for Website Fingerprinting," USENIX Security 2014 (the "kNN" / Wang dataset).

## What it is
Classic Tor WF closed-world dataset: **100 websites × ~100 undefended traces each**. Cell/packet
sequences. Widely reused as the "Wang" dataset in WF literature.

## How the paper uses it
"Hybrid CNN-SSM…" (IJCNN/WCCI 2026): single-tab closed-world evaluation, and the base for
generating **Tamaraw** and **CS-BuFLO** defended traces (those defended variants are produced
by the authors, not downloaded).

## Other datasets on the same Tao Wang page (not pulled)
`walkiebatch*.zip` (Walkie-Talkie — see `../WF-WalkieTalkie/`), `levdata*.zip`, `20000.zip`
(open-world), `defenses.zip`, `attacks.zip`.

## After downloading
`unzip knndata.zip`; record the trace format (each file = one visit; `site-instance`) below.
