#!/usr/bin/env bash
# Single confirmatory hold-out run (pre-registrations D13-D16), started once after tag `frozen-D16`.
# Step 1 (`15_holdout_eval.py --pre`) must already be committed. Three parallel streams, 4 threads each;
# refits are checkpointed one by one (cached_refit), so an interrupted stream is resumed by re-running
# the same command: finished refits are reloaded, never refitted with other settings.
# Step 3 (`15_holdout_eval.py --final`) runs once all streams have finished.
set -u
cd "$(dirname "$0")/.."
P=".venv/Scripts/python -W ignore -u"
mkdir -p logs
unset ECOPE_HOLDOUT_DRYRUN ECOPE_HOLDOUT_REP ECOPE_RESULTS_DIR
MODELS_PESTS="G_lgbm G_lgbm_nbr G_lgbm_nbr_leaky EcoPE-GAT_leaky EcoPE-GAT_causal EcoPE-GAT_asof EcoPE-GAT_asof_plus G_lgbm_nbr_asof"

( $P src/09_d13.py --holdout --threads 4 && echo "STREAM_A_OK" ) > logs/holdout_A.log 2>&1 &
( $P src/11_d14.py --holdout --threads 4 && $P src/11_d14.py --holdout --plus --threads 4 \
  && $P src/14_d16_lgbm_asof.py --holdout --threads 4 && echo "STREAM_B_OK" ) > logs/holdout_B.log 2>&1 &
( for c in P1_halys_europe P2_halys_namerica P3_popillia_namerica; do
    $P src/p02_pests_models.py "$c" --holdout --models $MODELS_PESTS --threads 4 || exit 1
  done && echo "STREAM_C_OK" ) > logs/holdout_C.log 2>&1 &
wait
grep -h "STREAM_.*_OK" logs/holdout_[ABC].log
if [ "$(grep -h "STREAM_.*_OK" logs/holdout_[ABC].log | wc -l)" -eq 3 ]; then
  $P src/15_holdout_eval.py --final > logs/holdout_eval.log 2>&1 && echo "EVAL_OK"
fi
