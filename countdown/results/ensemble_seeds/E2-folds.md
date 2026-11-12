# E2: ensemble vs single members across outer folds

5 outer folds at seed 42; the folds' test sets share no group. Test macro-F1. Deployed = soft vote over flow_gbdt + graph_gnn, temperature-calibrated. CV-selected = the combiner chosen on out-of-fold predictions. Best single is picked **on test** (favours the single side).

| outer fold | n test | flow_gbdt | best single | deployed | CV-selected |
| --- | --- | --- | --- | --- | --- |
| 0 | 9338 | 0.832 | 0.868 (graph_gnn) | 0.890 | 0.890 (vote/temperature) |
| 1 | 9359 | 0.821 | 0.870 (graph_gnn) | 0.891 | 0.891 (vote/temperature) |
| 2 | 9244 | 0.805 | 0.850 (graph_gnn) | 0.869 | 0.869 (vote/temperature) |
| 3 | 9241 | 0.824 | 0.860 (graph_gnn) | 0.882 | 0.882 (vote/temperature) |
| 4 | 9102 | 0.838 | 0.883 (graph_gnn) | 0.902 | 0.902 (vote/temperature) |

| | mean | sd | min | max |
| --- | --- | --- | --- | --- |
| flow_gbdt | 0.824 | 0.013 | 0.805 | 0.838 |
| best single | 0.866 | 0.012 | 0.850 | 0.883 |
| deployed | 0.887 | 0.013 | 0.869 | 0.902 |
| CV-selected | 0.887 | 0.013 | 0.869 | 0.902 |

| paired difference | mean | 95 % CI | wins / ties / losses | sign test p |
| --- | --- | --- | --- | --- |
| deployed − flow_gbdt | +0.063 | [+0.056, +0.070] | 5 / 0 / 0 | 0.062 |
| deployed − best single | +0.021 | [+0.019, +0.022] | 5 / 0 / 0 | 0.062 |
| CV-selected − best single | +0.021 | [+0.019, +0.022] | 5 / 0 / 0 | 0.062 |
