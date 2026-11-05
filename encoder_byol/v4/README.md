# BYOL anatomy v4 (ConvNeXt-T, RFMiD)

v4 = anatomy-aware BYOL (1 disc + 1 macula + 2 random views at 2.5–3.2 DD, weak colour jitter, light VICReg), 120 epochs on Kaggle T4×2.

- `RESULTS.md` — in-run results (v4's own fold).
- `phase0_rescore/RESULTS.md` — **re-score of ep20 on the locked A/B split** (fair comparison vs B). Headline: v4 ep20 target LP **0.921** vs B 0.911; rare ODC/ODP/ARMD 1/5/10 **0.589/0.688/0.725** vs B 0.622/0.705/0.734; ARMD@10 **0.943** vs B 0.969. Multi-region averaging hurts ARMD → use global (f_full) embedding only.
- `CONFIG_LOCKED.md`, `WORKFILE.md`, `argon/`, `argon_code_*` — design notes and reviews.
- `kernels/full`, `kernels/smoke` — Kaggle kernels; `build_kernel.py` builds them.
- `full_out/` — full-run logs, report/history/epoch evals, embeddings (`.npy`).
- `localtest/` — 1-epoch local smoke test outputs.

Phase-0 paths like `../full_out/out/byol_anatomy_v4/best.pth` refer to the release assets below (`.pth` files are not in git).

## Weights (GitHub Release `encoder-byol-v4`)
Checkpoint dicts follow the v1 format (`online`, `target_backbone`, ...). **Recommended encoder: `encoder_byol_anatomy_v4_best.pth` (epoch 20, `target_backbone`)** — the checkpoint scored in phase0_rescore. `last` = epoch 120. `localtest_*` are from the 1-epoch smoke test (not for use).

| file | size | sha256 |
|---|---|---|
| [encoder_byol_anatomy_v4_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_best.pth) | 244 MB | d685ae19152fa1e7540c5089b83ad2c16a451ad1605fc391d8549329c8a78278 |
| [encoder_byol_anatomy_v4_ep010.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_ep010.pth) | 244 MB | ab2e366de949229221bcdbd902ee5d0448e1eab7224bc822f9d4da40418f56f3 |
| [encoder_byol_anatomy_v4_ep020.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_ep020.pth) | 244 MB | b768d64ec238da1e212c4abde6b6d48406d27a697aaf5fd3ce0935b387fe712c |
| [encoder_byol_anatomy_v4_ep030.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_ep030.pth) | 244 MB | e7e0f045ba565d84b833512b0acac796062cbe4cdf27f2cf31591f9a5739e62e |
| [encoder_byol_anatomy_v4_ep040.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_ep040.pth) | 244 MB | 5bbf8c0160c13775a039bc02f4b9822cd239bd9463f9126caca36dc5a045d381 |
| [encoder_byol_anatomy_v4_ep050.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_ep050.pth) | 244 MB | 10b41f41f01fdc03ebff2ac0f565ddbbe9c58aabdcda8baeb7363ee6121bf1f6 |
| [encoder_byol_anatomy_v4_ep060.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_ep060.pth) | 244 MB | fd7e9cea75de3870e7d053036e4db661b6714b88d6142eee248f030920ee0ce9 |
| [encoder_byol_anatomy_v4_ep070.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_ep070.pth) | 244 MB | 9d771bae448382d860f7c268cd465ae72efa1d253bd480e196e72bd4fecffbc4 |
| [encoder_byol_anatomy_v4_ep080.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_ep080.pth) | 244 MB | 9b63b9e5f63ce346f8f4c8c2386ce055af8cb690adff2fd70bf6fd7c7de4ccad |
| [encoder_byol_anatomy_v4_ep090.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_ep090.pth) | 244 MB | 11a8fb51feb4ab0477757d076b39f6c499e0948ade93d2e160c565d1478bb8a6 |
| [encoder_byol_anatomy_v4_ep100.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_ep100.pth) | 244 MB | dc329e02ab61afef9938aa6444f8b50eececb9175a30bd9bf152044f917feade |
| [encoder_byol_anatomy_v4_ep110.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_ep110.pth) | 244 MB | 5e6f7c42811eebdecc5e8b2a6016b6d71f2f1aae1331577db2330e283a042876 |
| [encoder_byol_anatomy_v4_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_last.pth) | 244 MB | 7a249ce8c784bab844743a2053ee75ccae435bff80b526a8ef7030b23b24c9ad |
| [encoder_byol_anatomy_v4_localtest_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_localtest_best.pth) | 244 MB | 02b18547f28e77854ebb405489b87b59f67b432974bf87128649049c1fed82b3 |
| [encoder_byol_anatomy_v4_localtest_ep001.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_localtest_ep001.pth) | 244 MB | cb6f715bb57c1897e6b3519e7ce3773eca3dc02eaabeb845ba82f37815890633 |
| [encoder_byol_anatomy_v4_localtest_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v4/encoder_byol_anatomy_v4_localtest_last.pth) | 244 MB | 9828ce5dc96c979a9ed9ee9b9e34d938329d050142e7c01fd80f06823e43aa76 |
