# Cross-dataset generalisation (traffic_type, flow_gbdt)

Macro-F1 over the classes both sides contain; in-domain = the training side's own grouped 80/20 split on the same classes.

| shift | train → test | shared classes | in-domain | cross | drop | test flows (captures) |
| --- | --- | --- | --- | --- | --- | --- |
| corpus | ISCX-VPN → ISCX-Tor | 7 | 0.5764 | **0.1019** | 0.4745 | 28,432 (95) |
| corpus | ISCX-Tor → ISCX-VPN | 7 | 0.5781 | **0.1129** | 0.4652 | 16,094 (140) |
| tunnel | ISCX-VPN plain → ISCX-VPN tunnelled | 6 | 0.5978 | **0.3351** | 0.2627 | 2,365 (31) |
| tunnel | ISCX-VPN tunnelled → ISCX-VPN plain | 6 | 0.6814 | **0.1815** | 0.4999 | 13,474 (109) |
| tunnel | ISCX-Tor plain → ISCX-Tor tunnelled | 8 | 0.7766 | **0.138** | 0.6386 | 697 (51) |
| tunnel | ISCX-Tor tunnelled → ISCX-Tor plain | 8 | 0.6006 | **0.0781** | 0.5225 | 53,732 (44) |

Per-class F1, cross:

| train → test | browsing | email | chat | audio_streaming | video_streaming | file_transfer | voip | p2p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vpn->tor | · | 0.07 | 0.03 | 0.02 | 0.37 | 0.15 | 0.07 | 0.00 |
| tor->vpn | · | 0.07 | 0.07 | 0.05 | 0.38 | 0.01 | 0.16 | 0.05 |
| nonvpn->vpn | · | 0.69 | 0.09 | 0.00 | 0.36 | 0.35 | 0.52 | · |
| vpn->nonvpn | · | 0.29 | 0.01 | 0.05 | 0.10 | 0.05 | 0.59 | · |
| nontor->tor | 0.12 | 0.00 | 0.38 | 0.00 | 0.07 | 0.00 | 0.18 | 0.35 |
| tor->nontor | 0.01 | 0.00 | 0.06 | 0.01 | 0.02 | 0.00 | 0.00 | 0.52 |
