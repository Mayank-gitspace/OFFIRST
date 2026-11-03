# team_offirst lesion segmenter — 768 vs 1440 vs focal 1440 (Kaggle, Oct 31 2026)
Validation = 6 held-out IDRiD training images (seed 42, same in all runs). Official 27-image test NOT evaluated.

| Run | Best ep | MA | HE | EX | SE | mAUPR |
|---|---|---|---|---|---|---|
| v1 768 Dice+BCE | 9/15 | 0.484 | 0.792 | 0.860 | 0.696 | 0.708 |
| v2 1440 Dice+BCE | 11/19 (early stop) | 0.545 | 0.802 | 0.897 | 0.652 | 0.724 |
| v3 1440 focal+Dice | 9/17 (early stop) | 0.492 | 0.751 | 0.879 | 0.625 | 0.687 |

Dice @ best: v1 MA .509 HE .726 EX .786 SE .704; v2 .527/.742/.824/.691; v3 .518/.700/.815/.635
Focal vs Dice MA: best-epoch -0.053 (≥0.05); peak-over-epochs 0.546 vs 0.538 (-0.008).
SE collapse: v2 SE AUPR 0.013 (ep16), 0.027 (ep18); v3 ~0.01 and Dice 0 from ep10 on -> early stop.
Timing: frozen epochs ~20-24 s, unfrozen ~36-38 s (train+val); totals 628 s / 723 s / 908 s; v3 queued ~23 min.
Outputs: quick768/, hires1440/, focal1440/ (ckpt/best.pth, last.pth, history.json, log)
