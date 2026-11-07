#!/bin/bash
# anti-correlation sweep: design 1/2 x lam {.05,.2,.5,1}, each + repair phase
cd "$(dirname "$0")/.."
for d in 1 2; do
  for lam in 0.05 0.2 0.5 1.0; do
    short=$(echo $lam | sed 's/0\.//;s/\.//')
    case $lam in 0.05) short=005;; 0.2) short=02;; 0.5) short=05;; 1.0) short=10;; esac
    tag=ngL_ac_d${d}_l${short}
    echo "=== $tag (design $d, lam $lam) $(date +%H:%M:%S) ==="
    python scripts/run_anticorr.py --design $d --lam $lam \
        --epochs 100 --repair-epochs 8 --tag $tag \
        > runs/${tag}.log 2>&1 || echo "FAILED $tag"
  done
done
echo "SWEEP DONE $(date +%H:%M:%S)"
