# PQC triage

### iscxvpn

16,094 flows, 26,998.8 MB; handshake observable on 15.6% of flows; **exposed (at_risk + at_risk_ceiling): 15.6% of flows, 38.4% of bytes**.

| verdict | flows | share | bytes share |
| --- | --- | --- | --- |
| not_observable | 13,588 | 84.4% | 61.7% |
| at_risk | 1,554 | 9.7% | 28.2% |
| at_risk_ceiling | 952 | 5.9% | 10.1% |

Exposed flows by traffic_type (top 8):

| class | flows | share | bytes share |
| --- | --- | --- | --- |
| voip | 828 | 33.0% | 0.4% |
| video_streaming | 610 | 24.3% | 10.6% |
| file_transfer | 430 | 17.2% | 88.8% |
| email | 320 | 12.8% | 0.0% |
| chat | 250 | 10.0% | 0.1% |
| audio_streaming | 60 | 2.4% | 0.0% |
| p2p | 8 | 0.3% | 0.0% |

Exposed flows by readiness tier:

| tier | flows | share | bytes share |
| --- | --- | --- | --- |
| migrate_next | 1,078 | 43.0% | 0.6% |
| migrate_first | 750 | 29.9% | 88.8% |
| low_priority | 678 | 27.1% | 10.7% |

### iscxtor

54,429 flows, 22,740.1 MB; handshake observable on 10.7% of flows; **exposed (at_risk + at_risk_ceiling): 10.7% of flows, 9.9% of bytes**.

| verdict | flows | share | bytes share |
| --- | --- | --- | --- |
| not_observable | 48,581 | 89.3% | 90.1% |
| at_risk | 3,699 | 6.8% | 3.9% |
| at_risk_ceiling | 2,149 | 4.0% | 6.0% |

Exposed flows by traffic_type (top 8):

| class | flows | share | bytes share |
| --- | --- | --- | --- |
| browsing | 4,731 | 80.9% | 13.3% |
| video_streaming | 739 | 12.6% | 75.5% |
| voip | 145 | 2.5% | 0.3% |
| email | 140 | 2.4% | 8.8% |
| chat | 67 | 1.1% | 0.1% |
| audio_streaming | 12 | 0.2% | 1.9% |
| p2p | 12 | 0.2% | 0.1% |
| file_transfer | 2 | 0.0% | 0.0% |

Exposed flows by readiness tier:

| tier | flows | share | bytes share |
| --- | --- | --- | --- |
| migrate_next | 4,943 | 84.5% | 13.7% |
| low_priority | 763 | 13.1% | 77.5% |
| migrate_first | 142 | 2.4% | 8.8% |

### cstnet

46,372 flows, 10,690.3 MB; handshake observable on 99.8% of flows; **exposed (at_risk + at_risk_ceiling): 99.8% of flows, 99.7% of bytes**.

| verdict | flows | share | bytes share |
| --- | --- | --- | --- |
| at_risk | 46,289 | 99.8% | 99.7% |
| not_observable | 73 | 0.2% | 0.3% |
| unknown | 10 | 0.0% | 0.0% |

Exposed flows by app (top 8):

| class | flows | share | bytes share |
| --- | --- | --- | --- |
| amap.com | 500 | 1.1% | 0.0% |
| python.org | 500 | 1.1% | 0.0% |
| weibo.com | 500 | 1.1% | 1.8% |
| cloudfront.net | 498 | 1.1% | 0.0% |
| gmail.com | 498 | 1.1% | 0.0% |
| guancha.cn | 498 | 1.1% | 0.2% |
| jd.com | 498 | 1.1% | 0.1% |
| biligame.com | 497 | 1.1% | 1.7% |

### postquantumtls

15,759 flows, 4,679.1 MB; handshake observable on 82.6% of flows; **exposed (at_risk + at_risk_ceiling): 82.6% of flows, 86.4% of bytes**.

| verdict | flows | share | bytes share |
| --- | --- | --- | --- |
| at_risk | 10,443 | 66.3% | 79.4% |
| not_observable | 2,741 | 17.4% | 13.6% |
| at_risk_ceiling | 2,575 | 16.3% | 7.0% |

Exposed flows by app (top 8):

| class | flows | share | bytes share |
| --- | --- | --- | --- |
| com.king.candycrushsaga | 917 | 7.0% | 0.8% |
| com.kb.gearclicker | 652 | 5.0% | 0.8% |
| com.snakeattack.game | 475 | 3.6% | 0.9% |
| com.roblox.client | 450 | 3.5% | 1.6% |
| com.pranksounds.appglobaltd | 430 | 3.3% | 1.7% |
| com.VectorUpGames.TallManRun | 429 | 3.3% | 0.7% |
| com.kayac.DrawFight | 395 | 3.0% | 1.2% |
| com.makeovergame.studio | 385 | 3.0% | 1.3% |

### _pqc_synth (self-generated, through the pipeline)

3 flows, 0.0 MB; handshake observable on 100.0% of flows; **exposed (at_risk + at_risk_ceiling): 66.7% of flows, 64.6% of bytes**.

| verdict | flows | share | bytes share |
| --- | --- | --- | --- |
| at_risk | 2 | 66.7% | 64.6% |
| quantum_safe | 1 | 33.3% | 35.4% |

### Fidelity of the predicted breakdown (exposed test flows)

| expert | exposed test flows | accuracy | class-mix TV | worst class (gap) | tier-mix TV |
| --- | --- | --- | --- | --- | --- |
| E1 | 1,468 | 0.706 | 0.127 | file_transfer (-0.080) | 0.109 |
| E2 | 9,316 | 0.915 | 0.028 | nvidia.com (-0.003) | — |
