# CSTNET-TLS 1.3 Dataset

> ⚠️ **Manual download (Google Drive)** — hosted as Google Drive folders, which scripted
> `curl` can't pull reliably. Use a browser, or `gdown --folder <url>` (install: `pip install gdown`).

**Source:** ET-BERT repo (WWW'22) — https://github.com/linwhitehat/ET-BERT
Readme: `datasets/CSTNET-TLS 1.3/readme.md`

**Download links (Google Drive):**
- Dataset: https://drive.google.com/drive/folders/1JSsYmevkxQFanoKOi_i1ooA6pH3s9sDr?usp=sharing
- Fine-tuning split: https://drive.google.com/drive/folders/1KlZatGoNm-4qu04z0LfrTpZr2oDaHfzr?usp=sharing

**Paper:** Lin, X., Xiong, G., Gou, G., Li, Z., Shi, J., Yu, J., "ET-BERT: A Contextualized
Datagram Representation with Pre-training Transformers for Encrypted Traffic Classification,"
WWW 2022. DOI: 10.1145/3485447.3512217

## What it is
The **first TLS 1.3 traffic dataset** — collected March–July 2021 on the China Science and
Technology Network (CSTNET). **120 applications**; only anonymized data is released for privacy.
Website-fingerprinting task.

## Used by
- "Generalized Encrypted Traffic Classification Using Inter-Flow Signals," arXiv:2508.21558v1
  — website fingerprinting over TLS 1.3.

## Local state ✅
Present under `extracted/`:

| Path | What |
| --- | --- |
| `cstnet-tls 1.3/<domain>/*.pcap` | raw pcaps, one dir per class — **46,372 pcaps** |
| `cstnet-tls1.3/{train,valid,test}_dataset.tsv` | ET-BERT fine-tuning split, columns `label\ttext_a` |
| `cstnet-tls1.3/nolabel_test_dataset.tsv` | same test rows, `text_a` only (4,637) |
| `flow_500/x_{direction,len,message_type,time,datagram}_{train,valid,test}.npy` + `y_*` | flow-level arrays, ~500 samples/class |
| `packet_5000/x_datagram_{train,valid,test}.npy` + `y_*` | packet-level arrays, ~5000 samples/class |

**Split sizes (tsv):** train 37,097 · valid 4,638 · test 4,637 · **total 46,372**

**1 pcap → 1 flow.** Pcap count and tsv row count match exactly (46,372 = 46,372), so this release
carries no session grouping — only *per-flow* graphs are constructible here, not MAppGraph-style
inter-flow app-session graphs.

## Label mapping (120 classes) — verified
Labels `0–119` are the class directory names in **ASCII byte order**.

> ⚠️ **Use `sorted()` / `LC_ALL=C sort`, not a locale-aware sort.** Byte order puts `.` (0x2E)
> before `c` (0x63), so `51.la`=1 precedes `51cto.com`=2. A default-locale shell `sort` swaps
> these two and silently mislabels them. No other pair is affected.

Verified by matching per-class pcap counts against per-label tsv row counts across all three
splits: **120/120 exact, zero mismatches.**

Counts below are flows per class (= pcaps per class).

```
  0 163.com                 491     1 51.la                   139     2 51cto.com               433
  3 acm.org                 307     4 adobe.com               228     5 alibaba.com             306
  6 alicdn.com              485     7 alipay.com              469     8 amap.com                500
  9 amazonaws.com           476    10 ampproject.org          481    11 apple.com               494
 12 arxiv.org               429    13 asus.com                402    14 atlassian.net           159
 15 azureedge.net           496    16 baidu.com               484    17 bilibili.com            495
 18 biligame.com            498    19 booking.com             173    20 chia.net                 16
 21 chinatax.gov.cn         495    22 cisco.com               494    23 cloudflare.com          449
 24 cloudfront.net          498    25 cnblogs.com             496    26 codepen.io              331
 27 crazyegg.com            155    28 criteo.com              475    29 ctrip.com               492
 30 dailymotion.com         494    31 deepl.com               151    32 digitaloceanspaces.com  497
 33 duckduckgo.com          475    34 eastday.com             172    35 eastmoney.com           495
 36 elsevier.com            258    37 facebook.com            495    38 feishu.cn               482
 39 ggpht.com               495    40 github.com              493    41 gitlab.com              258
 42 gmail.com               498    43 goat.com                134    44 google.com              474
 45 grammarly.com           218    46 gravatar.com            494    47 guancha.cn              499
 48 huanqiu.com             399    49 huawei.com              138    50 hubspot.com             401
 51 huya.com                497    52 ibm.com                 174    53 icloud.com              491
 54 ieee.org                498    55 instagram.com           216    56 iqiyi.com               488
 57 jb51.net                463    58 jd.com                  498    59 kugou.com               485
 60 leetcode-cn.com         476    61 media.net               401    62 mi.com                  163
 63 microsoft.com           487    64 mozilla.org             494    65 msn.com                 479
 66 naver.com               497    67 netflix.com             418    68 nike.com                490
 69 notion.so               490    70 nvidia.com              399    71 office.net              397
 72 onlinedown.net          208    73 opera.com               368    74 oracle.com              179
 75 outbrain.com            191    76 overleaf.com            491    77 paypal.com              143
 78 pinduoduo.com           488    79 python.org              500    80 qcloud.com              497
 81 qq.com                  498    82 researchgate.net        407    83 runoob.com              443
 84 sciencedirect.com       153    85 semanticscholar.org     176    86 sina.com.cn             491
 87 smzdm.com               493    88 snapchat.com            489    89 sohu.com                495
 90 spring.io               160    91 springer.com            276    92 squarespace.com         119
 93 statcounter.com         260    94 steampowered.com        469    95 t.co                    496
 96 taboola.com             460    97 teads.tv                491    98 thepaper.cn             140
 99 tiktok.com              118   100 toutiao.com             487   101 twimg.com               495
102 twitter.com             284   103 unity3d.com             474   104 v2ex.com                135
105 vivo.com.cn             493   106 vk.com                  222   107 vmware.com              194
108 walmart.com             434   109 weibo.com               500   110 wikimedia.org           356
111 wikipedia.org           311   112 wp.com                  429   113 xiaomi.com              207
114 ximalaya.com            340   115 yahoo.com               493   116 yandex.ru               479
117 youtube.com             495   118 yy.com                  491   119 zhihu.com               487
```

**Imbalance:** capped at 500/class; median 474, mean 386, **min 16**.
Long left tail — `chia.net` 16, `tiktok.com` 118, `squarespace.com` 119, `goat.com` 134,
`v2ex.com` 135, `huawei.com` 138, `51.la` 139, `thepaper.cn` 140, `paypal.com` 143,
`deepl.com` 151; ~20 classes under 220. Since the headline metric is macro-F1, `chia.net`
(≈13/1/2 across the split) is effectively unlearnable and contributes outsized variance —
report macro-F1 with and without the sub-50 tail.

**Class character:** these are web services by domain/SNI, not behaviour categories. A large
share are CDN/tracker infrastructure that co-loads with a user-facing site
(`alicdn.com`/`alibaba.com`, `twimg.com`/`twitter.com`, `ggpht.com`/`youtube.com`,
`biligame.com`/`bilibili.com`) — the hard pairs, and where relational structure should help most.
