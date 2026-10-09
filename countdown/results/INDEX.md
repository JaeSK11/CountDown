# Results index

Copied from the gitignored `runs/` by `scripts/collect_results.py` (decision D4). Metrics, configs and split audits only -- no models, no checkpoints. The tables that interpret these live in `phases/phase4-models/`.

| group | run | macro-F1 | accuracy | n |
| --- | --- | --- | --- | --- |
| flow-stats/phase3 | `20260823-032104-flow_gbdt` | 0.6317 | 0.8837 | 17918 |
| flow-stats/phase3 | `20260823-032217-flow_gbdt` | 0.8983 | 0.9946 | 14045 |
| flow-stats/phase3 | `20260823-032359-flow_gbdt` | 0.4409 | 0.4891 | 92 |
| flow-stats/phase3 | `20260823-043052-flow_gbdt` | 0.8110 | 0.8376 | 9384 |
| flow-stats/replication-drapergil | `replicate_drapergil.json` |  |  |  |
| flow-stats/replication-drapergil | `ablate_features_vs_model.json` |  |  |  |
| flow-stats/replication-drapergil | `ablate_gbdt_unweighted.json` |  |  |  |
| flow-stats/classweight-retest | `classweight-retest/balanced-s1/20260917-173819-flow_gbdt` | 0.6273 | 0.8841 | 17918 |
| flow-stats/classweight-retest | `classweight-retest/balanced-s2/20260917-174231-flow_gbdt` | 0.6325 | 0.8851 | 17918 |
| flow-stats/classweight-retest | `classweight-retest/balanced-s42/20260917-172638-flow_gbdt` | 0.6317 | 0.8837 | 17918 |
| flow-stats/classweight-retest | `classweight-retest/unweighted-s1/20260917-174105-flow_gbdt` | 0.6417 | 0.8896 | 17918 |
| flow-stats/classweight-retest | `classweight-retest/unweighted-s2/20260917-174246-flow_gbdt` | 0.6451 | 0.8914 | 17918 |
| flow-stats/classweight-retest | `classweight-retest/unweighted-s42/20260917-173558-flow_gbdt` | 0.6474 | 0.8912 | 17918 |
| flow-stats/gbdt-bakeoff | `gbdt-bakeoff/cstnet-flow_catboost-s1/20260919-234851-flow_catboost` | 0.7599 | 0.7936 | 9384 |
| flow-stats/gbdt-bakeoff | `gbdt-bakeoff/cstnet-flow_catboost-s2/20260919-235612-flow_catboost` | 0.7563 | 0.7912 | 9384 |
| flow-stats/gbdt-bakeoff | `gbdt-bakeoff/cstnet-flow_catboost-s42/20260919-234132-flow_catboost` | 0.7615 | 0.7947 | 9384 |
| flow-stats/gbdt-bakeoff | `gbdt-bakeoff/cstnet-flow_gbdt-s1/20260919-234402-flow_gbdt` | 0.8108 | 0.8388 | 9384 |
| flow-stats/gbdt-bakeoff | `gbdt-bakeoff/cstnet-flow_gbdt-s2/20260919-235124-flow_gbdt` | 0.8092 | 0.8371 | 9384 |
| flow-stats/gbdt-bakeoff | `gbdt-bakeoff/cstnet-flow_gbdt-s42/20260919-233644-flow_gbdt` | 0.8124 | 0.8390 | 9384 |
| flow-stats/gbdt-bakeoff | `gbdt-bakeoff/iscx_pooled-flow_catboost-s1/20260919-233330-flow_catboost` | 0.6159 | 0.8770 | 17918 |
| flow-stats/gbdt-bakeoff | `gbdt-bakeoff/iscx_pooled-flow_catboost-s2/20260919-233415-flow_catboost` | 0.6126 | 0.8780 | 17918 |
| flow-stats/gbdt-bakeoff | `gbdt-bakeoff/iscx_pooled-flow_catboost-s42/20260919-233243-flow_catboost` | 0.6181 | 0.8764 | 17918 |
| flow-stats/gbdt-bakeoff | `gbdt-bakeoff/iscx_pooled-flow_gbdt-s1/20260919-233305-flow_gbdt` | 0.6417 | 0.8896 | 17918 |
| flow-stats/gbdt-bakeoff | `gbdt-bakeoff/iscx_pooled-flow_gbdt-s2/20260919-233351-flow_gbdt` | 0.6451 | 0.8914 | 17918 |
| flow-stats/gbdt-bakeoff | `gbdt-bakeoff/iscx_pooled-flow_gbdt-s42/20260919-233216-flow_gbdt` | 0.6474 | 0.8912 | 17918 |
| image/replication-okonkwo | `okonkwo/replication.json` |  |  |  |
| image/replication-okonkwo | `okonkwo/variant_flowpic1.json` |  |  |  |
| image/replication-okonkwo | `okonkwo/variant_flowpic3.json` |  |  |  |
| image/replication-okonkwo | `okonkwo/variant_resnet3.json` |  |  |  |
| image/seeds | `okonkwo/seeds/base_s1.json` |  |  |  |
| image/seeds | `okonkwo/seeds/base_s1_torch2.14cu130.json` |  |  |  |
| image/seeds | `okonkwo/seeds/base_s1_torch2.14cu130_20260916.json` |  |  |  |
| image/seeds | `okonkwo/seeds/base_s2.json` |  |  |  |
| image/seeds | `okonkwo/seeds/base_s3.json` |  |  |  |
| image/seeds | `okonkwo/seeds/base_s4.json` |  |  |  |
| image/seeds | `okonkwo/seeds/base_s5.json` |  |  |  |
| image/seeds | `okonkwo/seeds/base_t26_s1.json` |  |  |  |
| image/seeds | `okonkwo/seeds/base_t26_s2.json` |  |  |  |
| image/seeds | `okonkwo/seeds/base_t26_s3.json` |  |  |  |
| image/seeds | `okonkwo/seeds/base_t26_s4.json` |  |  |  |
| image/seeds | `okonkwo/seeds/base_t26_s5.json` |  |  |  |
| image/seeds | `okonkwo/seeds/fp3_s1.json` |  |  |  |
| image/seeds | `okonkwo/seeds/fp3_s2.json` |  |  |  |
| image/seeds | `okonkwo/seeds/fp3_s3.json` |  |  |  |
| image/seeds | `okonkwo/seeds/fp3_s4.json` |  |  |  |
| image/seeds | `okonkwo/seeds/fp3_s5.json` |  |  |  |
| image/seeds | `okonkwo/seeds/fp3_t26_s1.json` |  |  |  |
| image/seeds | `okonkwo/seeds/fp3_t26_s2.json` |  |  |  |
| image/seeds | `okonkwo/seeds/fp3_t26_s3.json` |  |  |  |
| image/seeds | `okonkwo/seeds/fp3_t26_s4.json` |  |  |  |
| image/seeds | `okonkwo/seeds/fp3_t26_s5.json` |  |  |  |
| image/seeds | `okonkwo/seeds/rec_s1.json` |  |  |  |
| image/seeds | `okonkwo/seeds/rec_s2.json` |  |  |  |
| image/seeds | `okonkwo/seeds/rec_s3.json` |  |  |  |
| image/seeds | `okonkwo/seeds/rec_s4.json` |  |  |  |
| image/seeds | `okonkwo/seeds/rec_s5.json` |  |  |  |
| image/mobileapp | `mobileapp-image/20260917-174312-flow_image_cnn_baseline` | 0.0041 | 0.0261 | 459 |
| image/mobileapp | `mobileapp-image/20260917-174552-flow_image_cnn_baseline` | 0.1649 | 0.2658 | 459 |
| image/mobileapp | `mobileapp-image-d5/baseline/20260920-002305-flow_image_cnn_baseline` | 0.1152 | 0.2353 | 459 |
| image/mobileapp | `mobileapp-image-d5/recommended-restored/20260920-002329-flow_image_cnn` | 0.2237 | 0.3529 | 459 |
| image/mobileapp | `mobileapp-image-d5/recommended/20260920-023527-flow_image_cnn` | 0.2636 | 0.3747 | 459 |
| sequence | `seq/iscx_pooled_traffic_type_seq/20260920-023502-seq_cnn` | 0.6158 | 0.8845 | 17918 |
| sequence | `seq/iscx_pooled_traffic_type_seq_baseline/20260920-002150-seq_cnn_baseline` | 0.5469 | 0.8206 | 17918 |
| sequence | `seq/mobileapp_activity_seq/20260920-010039-seq_cnn` | 0.2196 | 0.2935 | 92 |
| sequence | `seq/mobileapp_activity_seq_baseline/20260920-010034-seq_cnn_baseline` | 0.0733 | 0.1304 | 92 |
| bytes/replication-etbert | `etbert-repro/flow.json` | 0.8585 | 0.8736 | 4637 |
| bytes/replication-etbert | `etbert-repro/packet.json` | 0.9076 | 0.9066 | 58171 |
| bytes/replication-etbert | `etbert-repro-nopretrain/packet_scratch.json` | 0.6877 | 0.6842 | 58171 |
| bytes/scorecard | `byte-net/scorecard.json` |  |  |  |
| graph/replication-tfegnn | `tfegnn-random-tfegnn-refbytes/tor.json` | 0.9381 | 0.9601 |  |
| graph/replication-tfegnn | `tfegnn-random-tfegnn/tor.json` | 0.7996 | 0.9203 |  |
| graph/replication-tfegnn | `tfegnn-refbytes/tor.json` | 0.5641 | 0.7016 |  |
| graph/replication-tfegnn | `tfegnn/tor.json` | 0.3874 | 0.6230 |  |
| graph/tor | `graph-gnn/20260917-180352-graph_gnn_baseline` | 0.1759 | 0.1242 | 1393 |
| graph/tor | `graph-gnn/20260917-182032-graph_gnn_baseline` | 0.4756 | 0.6345 | 435 |
| graph/tor | `graph-gnn/20260917-183723-graph_gnn_baseline` | 0.4628 | 0.6651 | 415 |
| graph/tor | `graph-gnn/20260917-185414-graph_gnn_baseline` | 0.3647 | 0.4961 | 385 |
| graph/tor | `graph-gnn/20260917-191053-graph_gnn_baseline` | 0.5224 | 0.6453 | 375 |
| graph/tor | `graph-gnn/20260917-192514-graph_gnn_baseline` | 0.5258 | 0.6787 | 305 |
| graph/tor | `graph-gnn-noval/20260917-210435-graph_gnn_baseline` | 0.2969 | 0.1737 | 1393 |
| graph/tor | `graph-gnn-noval/20260917-211755-graph_gnn_baseline` | 0.3922 | 0.6023 | 435 |
| graph/tor | `graph-gnn-noval/20260917-213130-graph_gnn_baseline` | 0.4371 | 0.6530 | 415 |
| graph/tor | `graph-gnn-noval/20260917-214519-graph_gnn_baseline` | 0.4204 | 0.5844 | 385 |
| graph/tor | `graph-gnn-noval/20260917-215842-graph_gnn_baseline` | 0.6655 | 0.7387 | 375 |
| graph/tor | `graph-gin-tor/20260920-011935-graph_gnn` | 0.3062 | 0.1400 | 1393 |
| graph/tor | `graph-gin-tor/20260920-013410-graph_gnn` | 0.5329 | 0.6069 | 435 |
| graph/tor | `graph-gin-tor/20260920-014900-graph_gnn` | 0.5061 | 0.7036 | 415 |
| graph/tor | `graph-gin-tor/20260920-020351-graph_gnn` | 0.6273 | 0.6078 | 385 |
| graph/tor | `graph-gin-tor/20260920-021833-graph_gnn` | 0.7610 | 0.8427 | 375 |
| graph/tor | `graph-gin-tor/20260920-023354-graph_gnn` | 0.5839 | 0.6852 | 305 |
| graph/vpn | `graph-gnn-vpn/20260917-224131-graph_gnn_baseline` | 0.6222 | 0.7909 | 1827 |
| graph/vpn | `graph-gnn-vpn/20260917-224441-graph_gnn_baseline` | 0.4912 | 0.9690 | 5130 |
| graph/vpn | `graph-gnn-vpn/20260917-224800-graph_gnn_baseline` | 0.8926 | 0.9688 | 3849 |
| graph/vpn | `graph-gnn-vpn/20260917-225127-graph_gnn_baseline` | 0.8308 | 0.8499 | 2765 |
| graph/vpn | `graph-gnn-vpn/20260917-225437-graph_gnn_baseline` | 0.5415 | 0.7322 | 3349 |
| graph/vpn | `graph-gnn-vpn/20260917-225803-graph_gnn_baseline` | 0.8970 | 0.9277 | 3140 |
| graph/nontor | `graph-gnn-nontor/20260920-001833-graph_gnn_baseline` | 0.6444 | 0.8939 | 8865 |
| graph/nonvpn | `graph-gnn-nonvpn/20260920-105519-graph_gnn_baseline` | 0.3497 | 0.6920 | 29023 |
| graph/cstnet | `graph-cstnet/graph_gnn/20260920-035006-graph_gnn` | 0.8659 | 0.8915 | 9338 |
| graph/cstnet | `graph-cstnet/graph_gnn/20260920-055212-graph_gnn` | 0.8626 | 0.8902 | 9359 |
| graph/cstnet | `graph-cstnet/graph_gnn/20260920-062841-graph_gnn` | 0.8593 | 0.8855 | 9244 |
| graph/cstnet | `graph-cstnet/graph_gnn/20260920-070515-graph_gnn` | 0.8606 | 0.8870 | 9241 |
| graph/cstnet | `graph-cstnet/graph_gnn/20260920-074208-graph_gnn` | 0.8780 | 0.9011 | 9102 |
| graph/cstnet | `graph-cstnet/graph_gnn_baseline/20260920-033126-graph_gnn_baseline` | 0.8515 | 0.8775 | 9338 |
| graph/cstnet | `graph-cstnet/graph_gnn_baseline/20260920-053324-graph_gnn_baseline` | 0.7941 | 0.8337 | 9359 |
| graph/cstnet | `graph-cstnet/graph_gnn_baseline/20260920-060959-graph_gnn_baseline` | 0.8138 | 0.8435 | 9244 |
| graph/cstnet | `graph-cstnet/graph_gnn_baseline/20260920-064631-graph_gnn_baseline` | 0.8292 | 0.8566 | 9241 |
| graph/cstnet | `graph-cstnet/graph_gnn_baseline/20260920-072306-graph_gnn_baseline` | 0.8144 | 0.8415 | 9102 |
| phase5 | `ensemble/E1/20261008-180421/ablation.json` | 0.6648 | 0.8933 |  |
| phase5 | `ensemble/E2/20261008-195912/ablation.json` | 0.8900 | 0.9129 |  |
| phase5 | `ensemble/E3/20261008-181041/ablation.json` | 0.5134 | 0.5652 |  |
| phase5 | `ensemble/E1/20261008-180421/ensemble.json` |  |  |  |
| phase5 | `ensemble/E2/20261008-195912/ensemble.json` |  |  |  |
| phase5 | `ensemble/E3/20261008-181041/ensemble.json` |  |  |  |
| phase5 | `ensemble/E1/20261008-180421/routing.json` |  |  | 70523 |
| phase5 | `ensemble/E2/20261008-195912/routing.json` |  |  | 46372 |
| phase5 | `ensemble/E3/20261008-181041/routing.json` |  |  | 368 |
| phase5-data | `reparse/cstnet-20261009-005323.json` |  |  |  |
| phase5-data | `routing-eval/20261009-010228.json` |  |  |  |
| phase6 | `pipeline-skew/20261009-011430.json` |  |  |  |
