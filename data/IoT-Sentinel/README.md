# IoT-Sentinel Dataset (IoT device captures)

> ⚠️ **Manual/browser download required** — the host is behind a Cloudflare JS challenge that
> blocks scripted `curl`/`wget` (returns HTTP 403 `cf-mitigated: challenge`). Open the link in
> a real browser to fetch the ~25 MB zip.

**Source:** Aalto University research portal — https://research.aalto.fi/en/datasets/iot-devices-captures
**Direct file (browser only):** https://research.aalto.fi/files/13004478/captures_IoT_Sentinel.zip
**Size:** ~25.4 MB (`captures_IoT_Sentinel.zip`)
**Paper:** Miettinen, M., Marchal, S., Hafeez, I., Asokan, N., Sadeghi, A.-R., Tarkoma, S.,
"IoT Sentinel: Automated device-type identification for security enforcement in IoT," IEEE
ICDCS 2017. DOI: 10.1109/ICDCS.2017.283

## What it is
Network-traffic captures from **31 IoT devices** across connection technologies (WiFi, ZigBee,
Z-Wave) — smart lighting, home automation, security cameras, etc. Used for device-type
fingerprinting (the method extracts 23 packet features from the first 12 packets of a device's
setup traffic, working even for encrypted communication).

## Used by
- "Generalized Encrypted Traffic Classification Using Inter-Flow Signals," arXiv:2508.21558v1
  — IoT device fingerprinting evaluation.

## After downloading
Drop `captures_IoT_Sentinel.zip` here, `unzip`, and record the per-device pcap layout below.
