# Intrusion Detection Evaluation Dataset (ISCXIDS2012)

**Source:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB)
**Download page:** https://www.unb.ca/cic/datasets/ids.html (registration typically required)
**Total size:** ~84.4 GB across 7 days
**Formats:** Full packet payloads (`.pcap`) + relevant traffic profiles
**Local path:** `data/ISCXIDS2012/`

**Required citation:**
> Ali Shiravi, Hadi Shiravi, Mahbod Tavallaee, Ali A. Ghorbani, "Toward developing a
> systematic approach to generate benchmark datasets for intrusion detection," *Computers &
> Security*, Volume 31, Issue 3, May 2012, Pages 357–374, ISSN 0167-4048,
> https://doi.org/10.1016/j.cose.2011.12.012

---

## Motivation

In network intrusion detection (IDS), anomaly-based approaches in particular suffer from
accurate evaluation, comparison, and deployment, which originates from the scarcity of
adequate datasets. Many such datasets are internal and cannot be shared due to privacy
issues, others are heavily anonymized and do not reflect current trends, or they lack certain
statistical characteristics. These deficiencies are primarily why a perfect dataset is yet to
exist, forcing researchers to resort to often-suboptimal datasets.

As network behaviours and patterns change and intrusions evolve, it has become necessary to
move away from static, one-time datasets toward more **dynamically generated** datasets that
not only reflect the traffic compositions and intrusions of the time, but are also
**modifiable, extensible, and reproducible**.

## Approach — profiles

At ISCX, a systematic approach to generate the required datasets addresses this need. The
underlying notion is based on the concept of **profiles**, which contain detailed descriptions
of intrusions and abstract distribution models for applications, protocols, or lower-level
network entities. Real traces are analyzed to create profiles for agents that generate real
traffic for **HTTP, SMTP, SSH, IMAP, POP3, and FTP**. A set of guidelines outlines valid
datasets and sets the basis for generating profiles; these guidelines are vital for realism,
evaluation capabilities, total capture, completeness, and malicious activity.

Profiles are then employed in a testbed experiment to generate the dataset. Various
multi-stage attack scenarios were subsequently carried out to supply the anomalous portion.
The intent is to assist researchers in acquiring such datasets for testing, evaluation, and
comparison, through sharing the generated datasets and profiles.

To simulate user behaviour, the behaviours of the Center's users were abstracted into
profiles, and agents were programmed to execute them, effectively mimicking user activity.
Attack scenarios were designed and executed to express real-world malicious behaviour,
applied in real-time from physical devices via human assistance — thereby avoiding any
unintended characteristics of post-merging network attacks with real-time background traffic.
This arrangement allows the network traces to be **labeled**, simplifying IDS evaluation and
providing more realistic, comprehensive benchmarks.

## Dataset characteristics

- **Realistic network and traffic** — the dataset should not exhibit unintended properties
  (network- or traffic-wise), giving a clearer picture of the real effects of attacks and
  workstation responses. Both normal and anomalous traffic look and behave as realistically as
  possible; any artificial post-capture trace insertion is highly discouraged as it introduces
  inconsistencies.
- **Labeled dataset** — created in a controlled, deterministic environment that distinguishes
  anomalous activity from normal traffic, eliminating impractical manual labeling.
- **Total interaction capture** — includes all network interactions, either within or between
  internal LANs, providing the means to detect anomalous behaviour and correctly interpret
  results.
- **Complete capture** — generated in a controlled testbed, completely removing the need for
  sanitization/anonymization and preserving the naturalness of the data (full payloads
  retained).
- **Diverse intrusion scenarios** — a diverse set of multi-stage attacks, each carefully
  crafted toward recent security-threat trends (including service- and application-targeted
  attacks), not just traditional brute-force attempts.

The UNB ISCX IDS 2012 dataset consists of labeled network traces, including **full packet
payloads in pcap format**, which along with the relevant profiles are publicly available for
researchers.

## Contents — 7 days of network activity

| Day       | Date       | Description                                              | Size (GB) |
| --------- | ---------- | ------------------------------------------------------- | --------- |
| Friday    | 11/6/2010  | Normal Activity. No malicious activity                  | 16.1      |
| Saturday  | 12/6/2010  | Normal Activity. No malicious activity                  | 4.22      |
| Sunday    | 13/6/2010  | Infiltrating the network from inside + Normal Activity  | 3.95      |
| Monday    | 14/6/2010  | HTTP Denial of Service + Normal Activity                | 6.85      |
| Tuesday   | 15/6/2010  | Distributed Denial of Service using an IRC Botnet       | 23.4      |
| Wednesday | 16/6/2010  | Normal Activity. No malicious activity                  | 17.6      |
| Thursday  | 17/6/2010  | Brute Force SSH + Normal Activity                       | 12.3      |

_Total: ~84.4 GB._

## Used by

- "Encrypted traffic classification encoder based on lightweight graph representation,"
  *Scientific Reports* 2025 — used as the malicious-traffic-detection benchmark / main
  ablation. PDF: `reference/papers/Lightweight-graph-representation-...`

## After downloading

Place the per-day `.pcap` files (and profiles) here and update this section with the actual
file names and any train/test split you adopt.
