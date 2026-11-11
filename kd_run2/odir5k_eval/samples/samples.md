# ODIR-5K per-sample probe outputs

**Selection:** preferred (≥2 with A/G/M/D) (seed 7). Not purely random. A purely random draw of 5 test eyes with seed 7 included only 1 eye(s) with a disease label among A/G/M/D [1103_left.jpg (M), 4141_left.jpg (N), 3353_right.jpg (N), 534_right.jpg (N), 1034_right.jpg (N)]. With the same seed, 2 eyes were therefore drawn from the A/G/M/D-positive test subset and 3 from the remaining test eyes (final count with A/G/M/D: 3).

Protocol matches `evaluate.py`: frozen encoders, L2-normalised embeddings, patient-level 70/30 split from `embeddings/ids.csv`, `StandardScaler` fit on the train split, one-vs-rest `LogisticRegression(class_weight='balanced', max_iter=3000)` with the **clean-condition** `C` values already selected in `results.json['C_selected']`. Predicted values are class probabilities (`predict_proba`). Neighbours are the top-3 train eyes by cosine similarity on L2-normalised embeddings (clean train bank, including when the query is degraded). Encoders were not retrained. Degraded thumbnails use the kernel `degrade_eval` from `extract.py` (row index = `ids.csv` row, size 512).

C (clean) used:

| Model | N | D | G | C | A | H | M | O |
|---|---|---|---|---|---|---|---|---|
| KD 768 | 0.01 | 0.01 | 0.001 | 0.001 | 0.001 | 0.0001 | 0.001 | 0.01 |
| SSL 960 | 0.001 | 0.001 | 0.001 | 0.0001 | 0.001 | 0.001 | 0.01 | 0.001 |
| Teacher | 0.001 | 0.001 | 0.001 | 0.0001 | 0.001 | 0.001 | 0.001 | 0.001 |

## Sample 1: 1406_left.jpg

- Patient 1406, left eye, age 78, Female, test split
- True labels: `N=0, D=1, G=1, C=0, A=0, H=0, M=0, O=0`  →  **D+G (DR, Glaucoma)**
- Original ODIR diagnostic keywords: suspected glaucoma; mild nonproliferative retinopathy
- Montage: `<workdir>/kd_run2/odir5k_eval/samples/sample_01_1406_left.png`

### Clean query — predicted probabilities

| Label | True | KD 768 | SSL 960 | Teacher |
|---|---:|---:|---:|---:|
| N (Normal) | 0 | **0.640** | **0.767** | **0.843** |
| D (DR) | 1 | 0.146 | 0.200 | 0.155 |
| G (Glaucoma) | 1 | **0.910** | **0.842** | **0.872** |
| C (Cataract) | 0 | 0.006 | 0.046 | 0.094 |
| A (AMD) | 0 | 0.023 | 0.009 | 0.021 |
| H (Hypertension) | 0 | 0.260 | 0.167 | 0.069 |
| M (Myopia) | 0 | 0.019 | 0.003 | 0.165 |
| O (Other) | 0 | 0.446 | 0.224 | 0.364 |

### Clean query — top-3 train neighbours (cosine)

**KD 768**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 2846_left.jpg (patient 2846 left) | 0.701 | N | normal fundus |
| 2 | 153_left.jpg (patient 153 left) | 0.699 | G | glaucoma |
| 3 | 1474_left.jpg (patient 1474 left) | 0.698 | DG | glaucoma; moderate non proliferative retinopathy |

**SSL 960**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 1474_left.jpg (patient 1474 left) | 0.844 | DG | glaucoma; moderate non proliferative retinopathy |
| 2 | 153_left.jpg (patient 153 left) | 0.829 | G | glaucoma |
| 3 | 1478_right.jpg (patient 1478 right) | 0.815 | N | normal fundus |

**Teacher**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 4007_right.jpg (patient 4007 right) | 0.692 | D | mild nonproliferative retinopathy |
| 2 | 4192_right.jpg (patient 4192 right) | 0.641 | D | moderate non proliferative retinopathy |
| 3 | 153_left.jpg (patient 153 left) | 0.637 | G | glaucoma |

### Degraded query — predicted probabilities (clean-trained probe)

| Label | True | KD 768 | SSL 960 |
|---|---:|---:|---:|
| N (Normal) | 0 | **0.662** | **0.758** |
| D (DR) | 1 | 0.250 | 0.167 |
| G (Glaucoma) | 1 | **0.922** | **0.812** |
| C (Cataract) | 0 | 0.014 | 0.084 |
| A (AMD) | 0 | 0.014 | 0.008 |
| H (Hypertension) | 0 | 0.303 | 0.036 |
| M (Myopia) | 0 | 0.008 | 0.002 |
| O (Other) | 0 | 0.277 | 0.210 |

### Degraded query — top-3 train neighbours (clean train bank)

**KD 768**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 153_left.jpg (patient 153 left) | 0.711 | G | glaucoma |
| 2 | 2846_left.jpg (patient 2846 left) | 0.676 | N | normal fundus |
| 3 | 3028_right.jpg (patient 3028 right) | 0.675 | N | normal fundus |

**SSL 960**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 1236_left.jpg (patient 1236 left) | 0.785 | O | optic nerve atrophy |
| 2 | 153_left.jpg (patient 153 left) | 0.784 | G | glaucoma |
| 3 | 1291_left.jpg (patient 1291 left) | 0.761 | GO | glaucoma; macular epiretinal membrane |

## Sample 2: 4612_right.jpg

- Patient 4612, right eye, age 55, Female, test split
- True labels: `N=0, D=1, G=0, C=0, A=0, H=0, M=0, O=0`  →  **D (DR)**
- Original ODIR diagnostic keywords: moderate non proliferative retinopathy
- Montage: `<workdir>/kd_run2/odir5k_eval/samples/sample_02_4612_right.png`

### Clean query — predicted probabilities

| Label | True | KD 768 | SSL 960 | Teacher |
|---|---:|---:|---:|---:|
| N (Normal) | 0 | 0.366 | 0.309 | 0.277 |
| D (DR) | 1 | **0.757** | **0.760** | **0.897** |
| G (Glaucoma) | 0 | 0.226 | 0.073 | 0.377 |
| C (Cataract) | 0 | 0.010 | 0.026 | 0.062 |
| A (AMD) | 0 | 0.073 | 0.143 | 0.031 |
| H (Hypertension) | 0 | **0.864** | **0.813** | **0.845** |
| M (Myopia) | 0 | 0.002 | 0.000 | 0.002 |
| O (Other) | 0 | 0.090 | 0.212 | 0.218 |

### Clean query — top-3 train neighbours (cosine)

**KD 768**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 654_right.jpg (patient 654 right) | 0.754 | D | moderate non proliferative retinopathy |
| 2 | 2498_right.jpg (patient 2498 right) | 0.737 | N | normal fundus |
| 3 | 4631_right.jpg (patient 4631 right) | 0.731 | D | proliferative diabetic retinopathy |

**SSL 960**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 3944_right.jpg (patient 3944 right) | 0.882 | D | moderate non proliferative retinopathy |
| 2 | 213_left.jpg (patient 213 left) | 0.880 | N | normal fundus |
| 3 | 311_left.jpg (patient 311 left) | 0.874 | D | diabetic retinopathy |

**Teacher**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 4138_right.jpg (patient 4138 right) | 0.755 | D | mild nonproliferative retinopathy |
| 2 | 4270_right.jpg (patient 4270 right) | 0.744 | D | severe nonproliferative retinopathy |
| 3 | 825_left.jpg (patient 825 left) | 0.726 | D | moderate non proliferative retinopathy |

### Degraded query — predicted probabilities (clean-trained probe)

| Label | True | KD 768 | SSL 960 |
|---|---:|---:|---:|
| N (Normal) | 0 | 0.256 | 0.178 |
| D (DR) | 1 | **0.909** | **0.776** |
| G (Glaucoma) | 0 | 0.060 | 0.080 |
| C (Cataract) | 0 | 0.007 | 0.011 |
| A (AMD) | 0 | 0.035 | 0.007 |
| H (Hypertension) | 0 | **0.753** | **0.628** |
| M (Myopia) | 0 | 0.005 | 0.001 |
| O (Other) | 0 | 0.130 | 0.389 |

### Degraded query — top-3 train neighbours (clean train bank)

**KD 768**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 654_right.jpg (patient 654 right) | 0.685 | D | moderate non proliferative retinopathy |
| 2 | 4637_left.jpg (patient 4637 left) | 0.677 | N | normal fundus |
| 3 | 4576_right.jpg (patient 4576 right) | 0.675 | D | mild nonproliferative retinopathy |

**SSL 960**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 2435_right.jpg (patient 2435 right) | 0.785 | N | normal fundus |
| 2 | 4565_right.jpg (patient 4565 right) | 0.781 | D | moderate non proliferative retinopathy |
| 3 | 4581_left.jpg (patient 4581 left) | 0.778 | H | hypertensive retinopathy |

## Sample 3: 2372_right.jpg

- Patient 2372, right eye, age 80, Female, test split
- True labels: `N=1, D=0, G=0, C=0, A=0, H=0, M=0, O=0`  →  **N (Normal)**
- Original ODIR diagnostic keywords: normal fundus
- Montage: `<workdir>/kd_run2/odir5k_eval/samples/sample_03_2372_right.png`

### Clean query — predicted probabilities

| Label | True | KD 768 | SSL 960 | Teacher |
|---|---:|---:|---:|---:|
| N (Normal) | 1 | **0.710** | **0.771** | **0.703** |
| D (DR) | 0 | 0.307 | 0.149 | 0.087 |
| G (Glaucoma) | 0 | 0.449 | **0.521** | **0.694** |
| C (Cataract) | 0 | 0.022 | 0.123 | 0.128 |
| A (AMD) | 0 | 0.059 | 0.182 | 0.106 |
| H (Hypertension) | 0 | 0.171 | 0.030 | 0.066 |
| M (Myopia) | 0 | 0.066 | 0.003 | 0.184 |
| O (Other) | 0 | 0.350 | 0.374 | **0.575** |

### Clean query — top-3 train neighbours (cosine)

**KD 768**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 320_left.jpg (patient 320 left) | 0.779 | D | mild nonproliferative retinopathy |
| 2 | 3270_left.jpg (patient 3270 left) | 0.743 | N | normal fundus |
| 3 | 2961_left.jpg (patient 2961 left) | 0.740 | N | normal fundus |

**SSL 960**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 320_left.jpg (patient 320 left) | 0.870 | D | mild nonproliferative retinopathy |
| 2 | 3270_left.jpg (patient 3270 left) | 0.848 | N | normal fundus |
| 3 | 320_right.jpg (patient 320 right) | 0.843 | O | pigment epithelium proliferation |

**Teacher**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 2521_right.jpg (patient 2521 right) | 0.754 | N | normal fundus |
| 2 | 1507_left.jpg (patient 1507 left) | 0.743 | N | normal fundus |
| 3 | 3307_left.jpg (patient 3307 left) | 0.713 | N | normal fundus |

### Degraded query — predicted probabilities (clean-trained probe)

| Label | True | KD 768 | SSL 960 |
|---|---:|---:|---:|
| N (Normal) | 1 | **0.626** | **0.740** |
| D (DR) | 0 | 0.206 | 0.059 |
| G (Glaucoma) | 0 | **0.634** | **0.932** |
| C (Cataract) | 0 | 0.016 | 0.084 |
| A (AMD) | 0 | 0.098 | 0.303 |
| H (Hypertension) | 0 | 0.152 | 0.023 |
| M (Myopia) | 0 | 0.093 | 0.016 |
| O (Other) | 0 | 0.177 | 0.494 |

### Degraded query — top-3 train neighbours (clean train bank)

**KD 768**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 2961_left.jpg (patient 2961 left) | 0.720 | N | normal fundus |
| 2 | 2906_right.jpg (patient 2906 right) | 0.715 | N | normal fundus |
| 3 | 320_left.jpg (patient 320 left) | 0.694 | D | mild nonproliferative retinopathy |

**SSL 960**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 320_left.jpg (patient 320 left) | 0.789 | D | mild nonproliferative retinopathy |
| 2 | 3151_right.jpg (patient 3151 right) | 0.786 | N | normal fundus |
| 3 | 3270_left.jpg (patient 3270 left) | 0.778 | N | normal fundus |

## Sample 4: 2985_right.jpg

- Patient 2985, right eye, age 61, Female, test split
- True labels: `N=1, D=0, G=0, C=0, A=0, H=0, M=0, O=0`  →  **N (Normal)**
- Original ODIR diagnostic keywords: normal fundus
- Montage: `<workdir>/kd_run2/odir5k_eval/samples/sample_04_2985_right.png`

### Clean query — predicted probabilities

| Label | True | KD 768 | SSL 960 | Teacher |
|---|---:|---:|---:|---:|
| N (Normal) | 1 | **0.683** | **0.639** | **0.769** |
| D (DR) | 0 | 0.428 | 0.247 | 0.420 |
| G (Glaucoma) | 0 | 0.037 | 0.014 | 0.048 |
| C (Cataract) | 0 | 0.012 | 0.039 | 0.047 |
| A (AMD) | 0 | 0.062 | 0.474 | 0.065 |
| H (Hypertension) | 0 | 0.339 | 0.069 | 0.081 |
| M (Myopia) | 0 | 0.004 | 0.000 | 0.002 |
| O (Other) | 0 | 0.111 | 0.308 | 0.303 |

### Clean query — top-3 train neighbours (cosine)

**KD 768**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 377_right.jpg (patient 377 right) | 0.746 | A | dry age-related macular degeneration |
| 2 | 377_left.jpg (patient 377 left) | 0.715 | A | dry age-related macular degeneration |
| 3 | 149_right.jpg (patient 149 right) | 0.706 | O | epiretinal membrane |

**SSL 960**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 4500_right.jpg (patient 4500 right) | 0.783 | D | mild nonproliferative retinopathy |
| 2 | 377_right.jpg (patient 377 right) | 0.769 | A | dry age-related macular degeneration |
| 3 | 3115_right.jpg (patient 3115 right) | 0.762 | N | normal fundus |

**Teacher**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 4290_right.jpg (patient 4290 right) | 0.640 | N | normal fundus |
| 2 | 2630_right.jpg (patient 2630 right) | 0.636 | N | normal fundus |
| 3 | 2742_left.jpg (patient 2742 left) | 0.626 | N | normal fundus |

### Degraded query — predicted probabilities (clean-trained probe)

| Label | True | KD 768 | SSL 960 |
|---|---:|---:|---:|
| N (Normal) | 1 | **0.769** | **0.600** |
| D (DR) | 0 | 0.228 | 0.202 |
| G (Glaucoma) | 0 | 0.046 | 0.059 |
| C (Cataract) | 0 | 0.022 | 0.020 |
| A (AMD) | 0 | 0.030 | 0.140 |
| H (Hypertension) | 0 | 0.347 | 0.143 |
| M (Myopia) | 0 | 0.003 | 0.000 |
| O (Other) | 0 | 0.112 | **0.525** |

### Degraded query — top-3 train neighbours (clean train bank)

**KD 768**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 377_right.jpg (patient 377 right) | 0.714 | A | dry age-related macular degeneration |
| 2 | 2655_right.jpg (patient 2655 right) | 0.704 | N | normal fundus |
| 3 | 1066_right.jpg (patient 1066 right) | 0.684 | N | normal fundus |

**SSL 960**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 3115_right.jpg (patient 3115 right) | 0.689 | N | normal fundus |
| 2 | 1066_right.jpg (patient 1066 right) | 0.687 | N | normal fundus |
| 3 | 377_right.jpg (patient 377 right) | 0.676 | A | dry age-related macular degeneration |

## Sample 5: 1515_left.jpg

- Patient 1515, left eye, age 78, Male, test split
- True labels: `N=0, D=0, G=1, C=0, A=0, H=0, M=0, O=0`  →  **G (Glaucoma)**
- Original ODIR diagnostic keywords: glaucoma
- Montage: `<workdir>/kd_run2/odir5k_eval/samples/sample_05_1515_left.png`

### Clean query — predicted probabilities

| Label | True | KD 768 | SSL 960 | Teacher |
|---|---:|---:|---:|---:|
| N (Normal) | 0 | 0.106 | 0.218 | 0.139 |
| D (DR) | 0 | 0.062 | 0.088 | 0.129 |
| G (Glaucoma) | 1 | 0.341 | 0.402 | **0.525** |
| C (Cataract) | 0 | **0.967** | **0.945** | **0.944** |
| A (AMD) | 0 | 0.018 | 0.007 | 0.034 |
| H (Hypertension) | 0 | 0.130 | 0.051 | 0.189 |
| M (Myopia) | 0 | 0.113 | 0.026 | 0.035 |
| O (Other) | 0 | 0.477 | 0.427 | **0.523** |

### Clean query — top-3 train neighbours (cosine)

**KD 768**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 2170_right.jpg (patient 2170 right) | 0.763 | C | cataract |
| 2 | 2218_right.jpg (patient 2218 right) | 0.749 | C | cataract |
| 3 | 2198_right.jpg (patient 2198 right) | 0.747 | C | cataract |

**SSL 960**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 2134_right.jpg (patient 2134 right) | 0.862 | C | cataract |
| 2 | 1285_left.jpg (patient 1285 left) | 0.850 | C | cataract |
| 3 | 2116_right.jpg (patient 2116 right) | 0.845 | C | cataract |

**Teacher**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 2204_left.jpg (patient 2204 left) | 0.760 | C | cataract |
| 2 | 2234_left.jpg (patient 2234 left) | 0.756 | C | cataract |
| 3 | 3146_left.jpg (patient 3146 left) | 0.735 | N | normal fundus |

### Degraded query — predicted probabilities (clean-trained probe)

| Label | True | KD 768 | SSL 960 |
|---|---:|---:|---:|
| N (Normal) | 0 | 0.251 | 0.337 |
| D (DR) | 0 | 0.019 | 0.041 |
| G (Glaucoma) | 1 | 0.154 | 0.455 |
| C (Cataract) | 0 | **0.976** | **0.915** |
| A (AMD) | 0 | 0.027 | 0.014 |
| H (Hypertension) | 0 | 0.134 | 0.048 |
| M (Myopia) | 0 | 0.043 | 0.004 |
| O (Other) | 0 | 0.446 | 0.327 |

### Degraded query — top-3 train neighbours (clean train bank)

**KD 768**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 2170_right.jpg (patient 2170 right) | 0.777 | C | cataract |
| 2 | 2204_left.jpg (patient 2204 left) | 0.733 | C | cataract |
| 3 | 2218_right.jpg (patient 2218 right) | 0.730 | C | cataract |

**SSL 960**

| Rank | Train neighbour | Cosine | Labels | Keywords |
|---:|---|---:|---|---|
| 1 | 1405_right.jpg (patient 1405 right) | 0.766 | O | refractive media opacity |
| 2 | 2240_right.jpg (patient 2240 right) | 0.745 | C | cataract |
| 3 | 2116_left.jpg (patient 2116 left) | 0.738 | C | cataract |

## PNG paths

- `<workdir>/kd_run2/odir5k_eval/samples/sample_01_1406_left.png`
- `<workdir>/kd_run2/odir5k_eval/samples/sample_02_4612_right.png`
- `<workdir>/kd_run2/odir5k_eval/samples/sample_03_2372_right.png`
- `<workdir>/kd_run2/odir5k_eval/samples/sample_04_2985_right.png`
- `<workdir>/kd_run2/odir5k_eval/samples/sample_05_1515_left.png`
