#!/bin/bash
# After the FULL KD run: download outputs to ../full_out, then run this (CPU, box). B = KD 768 (after BN recal).
# A1 = SSL ep40 target 960-d (SSL arm's own embeddings, as reported in RESULTS.md); A2 = SSL ep40 Procrustes-768 (from the KD kernel).
set -e; cd "$(dirname "$0")"; K=../full_out/out/kd_run2; S=encoder_mnv3_ssl/full_out/out/byol_mnv3_anatomy_v4
PY=/workspace/.evalvenv/bin/python
cmp <(cut -d, -f1 $K/embedding_ids.csv) <(cut -d, -f1 $S/embedding_ids.csv) && echo "ID order identical"
$PY paired_bootstrap.py --a $S/embeddings_target.npy --a-deg $S/embeddings_target_degraded.npy --b $K/embeddings_kd768.npy --b-deg $K/embeddings_kd768_degraded.npy --out kd768_vs_ssl960.json | tee kd768_vs_ssl960.log
$PY paired_bootstrap.py --a $K/embeddings_ssl_ep40_768procrustes.npy --a-deg $K/embeddings_ssl_ep40_768procrustes_degraded.npy --b $K/embeddings_kd768.npy --b-deg $K/embeddings_kd768_degraded.npy --out kd768_vs_ssl768p.json | tee kd768_vs_ssl768p.log
$PY paired_bootstrap.py --a $S/embeddings_target.npy --a-deg $S/embeddings_target_degraded.npy --b $K/embeddings_ssl_ep40_960.npy --b-deg $K/embeddings_ssl_ep40_960_degraded.npy --out sanity_ssl960_box_vs_kernel.json | tee sanity_ssl960_box_vs_kernel.log
