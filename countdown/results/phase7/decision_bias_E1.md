# Decision bias, E1 (iscx_pooled)

Fit on out-of-fold predictions of the train split (52605 flows); objective macro-F1 (flows), coordinate ascent, grid -3..3 step 0.25.

| class | bias |
| --- | --- |
| browsing | -1.16 |
| email | +0.09 |
| chat | +0.34 |
| audio_streaming | -0.91 |
| video_streaming | +0.34 |
| file_transfer | +0.59 |
| voip | +0.59 |
| p2p | +0.09 |

| split | metric | before | after |
| --- | --- | --- | --- |
| oof | macro_f1 | 0.5901 | 0.6166 |
| oof | macro_f1_bytes | 0.526 | 0.5649 |
| oof | file_transfer_recall | 0.049 | 0.2049 |
| oof | file_transfer_recall_bytes | 0.6363 | 0.8058 |
| oof | class_mix_tv | 0.08 | 0.0778 |
| oof | class_mix_tv_bytes | 0.2772 | 0.173 |
| oof | file_transfer_gap_pts | -6.78 | -4.71 |
| oof | file_transfer_gap_pts_bytes | -16.18 | -4.85 |
| test | macro_f1 | 0.6654 | 0.6747 |
| test | macro_f1_bytes | 0.7548 | 0.7727 |
| test | file_transfer_recall | 0.4663 | 0.4935 |
| test | file_transfer_recall_bytes | 0.9555 | 0.9632 |
| test | class_mix_tv | 0.0164 | 0.0228 |
| test | class_mix_tv_bytes | 0.0747 | 0.0748 |
| test | file_transfer_gap_pts | -0.43 | 0.01 |
| test | file_transfer_gap_pts_bytes | 1.09 | 1.72 |
