#!/bin/bash
# Full label_ratio sweep across all 4 datasets, run SEQUENTIALLY:
#   ctu13 -> unswnb15 -> cicids2017 -> cicids2018
# For each dataset, runs label_ratio = 5,10,20,30,40,50,60,70,80 (%) in order,
# each via that dataset's own *_spider_owl_main2.py 3-seed wrapper (which in
# turn calls the fixed *_2.py script -- originals are never touched).
#
# Output: one combined doc per dataset in ./label_ratio_reports/, named
# <dataset>_label_ratio.txt, with each ratio's clean report appended under a
# "label_ratio=XX%" header, in the format:
#
#   label_ratio=40%
#   -------------------
#   <argument table, self-label stats, per-task/per-seed tables, etc.>
#
# LIVE PROGRESS: while running, this script prints which dataset / label_ratio
# / seed is currently in flight to the TERMINAL (stderr) as it goes -- this is
# kept separate from the doc files, which only ever receive the clean report
# (stdout of each *_main2.py run). The underlying per-epoch training noise
# (tqdm bars, per-batch losses) goes to neither -- it's captured per-seed in
# training_logs/<ds>_seed<seed>_lr<ratio>.log for debugging if something looks
# wrong, but doesn't clutter either the terminal or the doc.
#
# WARNING: this is a LOT of compute. Each *_main2.py call runs 3 seeds
# in sequence; some datasets are far slower than others (cicids2018 alone
# took 3.5+ hours for a SINGLE seed at a single ratio during validation).
# Running all 9 ratios x 4 datasets sequentially (108 individual training
# runs total) could take many hours to multiple days depending on hardware.
# Meant to be started and left running (e.g. under nohup/tmux/screen), not
# run interactively start-to-finish.
#
# Usage:
#   bash run_full_label_ratio_sweep.sh
#   nohup bash run_full_label_ratio_sweep.sh > sweep_progress.log 2>&1 &
#   (with nohup as above, live progress lands in sweep_progress.log instead
#   of the terminal -- tail -f sweep_progress.log to watch it)

set -u
cd "$(dirname "$0")"

PY=/home/suresh/anaconda3/envs/temp/bin/python
OUTDIR=./label_ratio_reports
mkdir -p "$OUTDIR"

# fraction -> percentage label, in the order they should run
RATIOS=(0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80)
LABELS=(5 10 20 30 40 50 60 70 80)

run_dataset () {
    local dataset_name="$1"
    local main_script="$2"
    local outfile="$OUTDIR/${dataset_name}_label_ratio.txt"

    echo ""
    echo "=========================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting dataset: $dataset_name -> $outfile"
    echo "=========================================================="

    : > "$outfile"  # fresh file for this dataset

    for i in "${!RATIOS[@]}"; do
        local ratio="${RATIOS[$i]}"
        local pct="${LABELS[$i]}"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $dataset_name: label_ratio=${pct}% starting..."

        # Only stdout goes to the doc (the clean report); stderr (this script's
        # own progress lines plus each *_main2.py's per-seed [PROGRESS] prints)
        # flows through to the terminal/nohup log instead, live.
        {
            echo "label_ratio=${pct}%"
            echo "-------------------"
            echo ""
            "$PY" "$main_script" --label_ratio="$ratio"
            echo ""
        } >> "$outfile"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $dataset_name: label_ratio=${pct}% done."
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished dataset: $dataset_name"
}

run_dataset "ctu13"       "ctu13_spider_owl_main2.py"
run_dataset "unswnb15"    "unswnb15_spider_owl_main2.py"
run_dataset "cicids2017"  "cicids2017_spider_owl_main2.py"
run_dataset "cicids2018"  "cicids2018_spider_owl_main2.py"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL EXPERIMENTS COMPLETE. Reports in $OUTDIR/"
