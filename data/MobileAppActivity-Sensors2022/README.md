# Mobile-App In-App Activity Dataset (Sensors 2022) — ✅ DOWNLOADED & EXTRACTED

**Status:** Pulled and extracted. `Dataset.7z` (1.87 MB) → `D-OutCsv/` (368 CSVs, ~35 MB).
**Source:** Authors' Dropbox (open) — https://www.dropbox.com/s/9tihcj9wx2sia1t/Dataset.7z?dl=0
**Paper:** "Deep Learning for Encrypted Traffic Classification and Unknown Data Detection,"
*Sensors*, 22(19):7643, 2022. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9570541/

## What it is
Self-collected encrypted-traffic dataset of **fine-grained in-app activities** for **8 mobile
apps**. Traffic captured over WiFi with `Airmon-ng` / `Airodump-ng`, one target app at a time
(no background apps). Each activity was recorded **4 times**.

- **8 apps · 92 activities · 368 CSV capture files** (= 92 × 4 repetitions).
- The paper derives windowed samples from these raw captures at four window sizes:
  0.5 s (65,189 samples), 0.2 s (231,824), 0.05 s (1,876,624), 0.02 s (7,067,363).
  Best validation accuracy (~95%) reported at the 0.2 s window.

## Layout
```
D-OutCsv/<App>/<Activity>/<sample>.csv
```

| App | Activities | CSV files |
| --- | --- | --- |
| FB (Facebook) | 22 | 88 |
| Instagram | 20 | 80 |
| Messenger | 10 | 40 |
| Viber | 9 | 36 |
| WhatsApp | 9 | 36 |
| YouTube | 9 | 36 |
| Skype | 8 | 32 |
| Gmail | 5 | 20 |
| **Total** | **92** | **368** |

## CSV format (frame-level, from Wireshark/tshark)
```
frame.number, frame.time_relative, wlan.sa, wlan.da, wlan.ta, wlan.ra,
frame.time_delta_displayed, frame.len
```
Each row = one 802.11 frame: index, relative time, source/dest/transmitter/receiver MAC
addresses, inter-frame delta, and frame length. Sample/activity labels come from the folder
path (app → activity). Note the traffic is **encrypted**; features are metadata (timing +
size + MAC), not payload.

## Used by
- The Sensors 2022 paper above (primary train/test set + unknown-data detection).

## Notes
- Unlike most sets in this `data/` folder, this one is **openly and directly downloadable**.
- Raw `Dataset.7z` kept in-folder for re-extraction (`7z x Dataset.7z`).
