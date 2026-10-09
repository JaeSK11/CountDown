# Phase-5 ensemble ablation (per target)

Macro-F1 on each expert's grouped test split, seed 42. `selected` = combiner chosen by cross-validation on out-of-fold predictions (never by test).

| expert | target | best single | selected ensemble | deployed members | verdict |
| --- | --- | --- | --- | --- | --- |
| E1 | traffic_type | flow_gbdt 0.6654 | flow_gbdt + seq_cnn + flow_image_cnn (vote, temperature) 0.6648 | flow_gbdt | ✗ |
| E2 | app | graph_gnn 0.8680 | flow_gbdt + graph_gnn (vote, temperature) 0.8900 | flow_gbdt + graph_gnn | ✓ |
| E3 | activity | flow_gbdt 0.4409 | flow_gbdt + seq_cnn + flow_image_cnn (vote, temperature) 0.5134 | flow_gbdt + seq_cnn + flow_image_cnn | ✓ |
