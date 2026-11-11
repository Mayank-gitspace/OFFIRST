#!/bin/bash
# final-build local tests: (1) RUN_MODE=eval on the smoke best.pth (full 1,920 rows, refs on), (2) smoke re-run on the final build
cd encoder_mnv3_ssl
export HF_HOME=/workspace/hfcache TOB_INPUT=/workspace/rfmid_train_dl TOB_TMP=/workspace/byol_local/tmp_mnv3 TOB_WORKERS=4 TOB_PRETRAINED=1
cp localtest/out/byol_anatomy_v4/best.pth /tmp/mnv3_smoke_best.pth
rm -rf localtest/out_eval; TOB_RUN_MODE=eval TOB_CKPT=/tmp/mnv3_smoke_best.pth TOB_OUT=$PWD/localtest/out_eval \
  /workspace/kenv/bin/python kernels/full/team_offirst_encoder_byol_mnv3.py > localtest/eval_mode.log 2>&1; echo "eval exit $?" >> localtest/eval_mode.log
mv localtest/out localtest/out_smoke_prefix; mv localtest/smoke.log localtest/out_smoke_prefix/smoke_console.log
TOB_OUT=$PWD/localtest/out TOB_SMOKE_N=32 TOB_SMOKE_EPOCHS=2 TOB_BS=4 TOB_EVAL_SUB=150 TOB_EVAL_EPOCHS=1,2 TOB_REFS_IN_SMOKE=1 \
  /workspace/kenv/bin/python kernels/smoke/team_offirst_encoder_byol_mnv3.py > localtest/smoke.log 2>&1; echo "smoke exit $?" >> localtest/smoke.log
