#!/bin/zsh
# Every experiment in its own process (two BLAS threads each), then the live-session check, then figures and tables.
cd "$(dirname $0)/../.."
export VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=.
for e in main ablations curve lofo transfer few_shot robust calibration ood; do
  .venv/bin/python -W ignore experiments/paper/run_experiments.py $e > experiments/paper/out/exp_$e.log 2>&1 &
done
.venv/bin/python -W ignore experiments/paper/check_equivalence.py 3 > experiments/paper/out/exp_equivalence.log 2>&1 &
wait
echo "all experiments finished"
