import subprocess
import os
import sys
import tempfile
import json
import argparse
import time
from tabulate import tabulate
import numpy as np
import warnings





if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    start_time=time.time()
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)') 
    parser.add_argument('--ds', type=str, default="ids17", metavar='S',help='dataset name')
    parser.add_argument('--b_m', type=float, default=0.2, metavar='S',help='batch memory ratio(default: 0.2)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='S',help='learning rate(default: 0.001)')
    parser.add_argument('--wd', type=float, default=1e-3, metavar='S',help='weight decay(default: 0.01)')
    parser.add_argument('--label_ratio', type=float, default=0.2, metavar='S',help='labeled ratio (default: 0.1)')
    parser.add_argument('--nps', type=int, metavar='S',default=10000,help='number of projection samples(default: 100)')
    parser.add_argument('--bma', type=float, metavar='S',default=0.8,help='batch minority allocation(default: 0)')
    parser.add_argument('--alpha', type=float, metavar='S',default=1,help='distill loss multiplier(default: 0)')
    parser.add_argument('--lab_samp_in_mem_ratio', type=float, metavar='S',default=1.0,help='Percentage of labeled samples to store in memory(default: 1.0)')
    parser.add_argument('--bool_gpm', type=str, metavar='S',default="True",help='Enables gradient projections(default: True)')
    parser.add_argument('--mem_strat', type=str, metavar='S',default="equal",help='Buffer memory strategy(default: full initialization)')
    parser.add_argument('--training_cutoff', type=int, default=2, metavar='S',help='train the model for first n tasks and test for time decay on the rest')
    parser.add_argument('--bool_closs', type=str, metavar='S',default="False",help='Enables using contrastive loss(default: False)')
    parser.add_argument('--mlps', type=int, metavar='S',default=1,help='Number of learners (MLPs)default: 1)')
    parser.add_argument('--cos_dist', type=float, metavar='S',default=0.3,help='cosine distance for OWL(default: 0.3)')
    parser.add_argument('--mode_val', type=int, metavar='S',default=99,help='Mode value for OWL (default: 99)')

    args = parser.parse_args()
    auc_results = {}
    seed_list = [1,2,3]
    # seed_list = [1323,2323232,332323232]
    # seed_list = [1]
    curr_dir = os.getcwd()
    for seed_value in seed_list:
        print(f"[PROGRESS] dataset={args.ds}  label_ratio={args.label_ratio}  seed={seed_value}  starting...", file=sys.stderr, flush=True)
        fd, temp_file_name = tempfile.mkstemp() # create temporary file
        
        os.close(fd) # close the file
        cmd = ("/home/suresh/anaconda3/envs/temp/bin/python cicids2017_spider_owl_neurips2024_2.py --seed="+str(seed_value)
               +" --ds="+str(args.ds)+" --lr="+str(args.lr)+" --wd="+str(args.wd)
               +" --alpha="+str(args.alpha)+" --gpu="+str(args.gpu)
               +" --filename="+str(temp_file_name)+" --cos_dist="+str(args.cos_dist)
               +" --mode_val="+str(args.mode_val)+" --bool_gpm="+str(args.bool_gpm)
               +" --mem_strat="+str(args.mem_strat)+" --b_m="+str(args.b_m)
               +" --label_ratio="+str(args.label_ratio)
               +" --lab_samp_in_mem_ratio="+str(args.lab_samp_in_mem_ratio)
               +" --nps="+str(args.nps)+" --bma="+str(args.bma)
               +" --training_cutoff="+str(args.training_cutoff)
               +" --bool_closs="+str(args.bool_closs)+" --mlps="+str(args.mlps))
        # Route the underlying training run's raw stdout/stderr (per-epoch
        # losses, tqdm bars, self-labeling prints, etc.) to a log file instead
        # of this wrapper's own stdout -- keeps the aggregate report clean.
        os.makedirs('training_logs', exist_ok=True)
        train_log_path = os.path.join('training_logs', f'{args.ds}_seed{seed_value}_lr{args.label_ratio}.log')
        with open(train_log_path, 'w') as train_log_fp:
            proc = subprocess.Popen([cmd],shell=True,cwd=curr_dir,stdout=train_log_fp,stderr=subprocess.STDOUT)
            proc.communicate()
        with open(temp_file_name) as fp:
            result = json.load(fp)
            # auc_results[str(seed_value)] = result#result[str(seed_value)]
            auc_results[str(seed_value)] = result[str(seed_value)]
        os.unlink(temp_file_name)    

    print("{:<20}  {:<20}".format('Argument','Value'))
    print("*"*80)
    for arg in vars(args):
        print("{:<20}  {:<20}".format(arg, getattr(args, arg)))
    print("*"*80)    
    # print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20} {:<10}".format('seed','PR-AUC(O)', 'PR-AUC(I)', 'ROC-AUC','grad_norm_mean','grad_norm_variance'))
    print("*"*80)

    # ── Collect OWL labeling counts from each seed run ──────────────────────────
    # value[2..5] = str counts: self_label_benign, self_label_attack,
    #                            analyst_label_benign, analyst_label_attack
    label_stats = {}
    for key, value in auc_results.items():
        label_stats[key] = [float(value[2]), float(value[3]),
                            float(value[4]), float(value[5])]

    # Compute mean and std of labeling counts across seeds
    label_arr = np.array(list(label_stats.values()))  # shape: (n_seeds, 4)
    label_avg = label_arr.mean(axis=0)
    label_std = label_arr.std(axis=0)

    # Print labeling stats table (same format as before)
    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format(
        'Cols', 'Self_labels (Benign)', 'Self_labels (Attack)', 'Total (self-label)',
        'Analyst_labels (Benign)', 'Analyst_labels (Attack)', 'Total (analyst-label)'))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format(
        'Mean',
        label_avg[0], label_avg[1], label_avg[0] + label_avg[1],
        label_avg[2], label_avg[3], label_avg[2] + label_avg[3]))
    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format(
        'Variance',
        label_std[0], label_std[1], '---',
        label_std[2], label_std[3], '---'))
    print("-"*80)

    # ── Helper: AUT via trapezoidal rule over per-task PR-AUC values ─────────────
    # AUT = (1/(N-1)) * sum of trapezoids between consecutive tasks.
    # This is computed on the seed-averaged curve, not averaged over per-seed AUTs.
    def compute_aut(prauc_list):
        n = len(prauc_list)
        if n < 2:
            return float('nan')
        return sum((prauc_list[i] + prauc_list[i + 1]) / 2 for i in range(n - 1)) / (n - 1)

    # ── Per-task PR-AUC, FPR, FNR averaged across seeds + AUT on averaged curve ──
    # result_key mapping:
    #   0 = seen tasks result   (tasks 0 .. training_cutoff-1)
    #   1 = unseen tasks result (tasks training_cutoff .. end)
    #   6 = all tasks result    (tasks 0 .. end)
    #
    # Each result has the structure returned by testing():
    #   [0] prauc_in_pnt  – list of PR-AUC (benign) per task
    #   [1] prauc_out_pnt – list of PR-AUC (attack) per task
    #   [7] fpr_pnt       – list of FPR per task
    #   [8] fnr_pnt       – list of FNR per task
    def _per_task_stats(result_key, split_label):
        # Step 1: gather per-seed per-task lists
        prauc_ben_seeds, prauc_att_seeds, fpr_seeds, fnr_seeds = [], [], [], []
        for seed_res in auc_results.values():
            r = seed_res[result_key]
            prauc_ben_seeds.append(r[0])  # PR-AUC benign list
            prauc_att_seeds.append(r[1])  # PR-AUC attack list
            fpr_seeds.append(r[7])        # FPR list
            fnr_seeds.append(r[8])        # FNR list

        # Step 2: build (n_seeds, n_tasks) arrays; skip on mismatch
        try:
            pb = np.array(prauc_ben_seeds)  # shape: (n_seeds, n_tasks)
            pa = np.array(prauc_att_seeds)
            fr = np.array(fpr_seeds)
            fn = np.array(fnr_seeds)
        except ValueError:
            print(f"  Skipping {split_label}: inconsistent task counts across seeds")
            return

        n_tasks = pb.shape[1]

        # Step 3: mean and std per task across seeds
        pb_mean, pb_std = pb.mean(axis=0), pb.std(axis=0)
        pa_mean, pa_std = pa.mean(axis=0), pa.std(axis=0)
        fr_mean, fr_std = fr.mean(axis=0), fr.std(axis=0)
        fn_mean, fn_std = fn.mean(axis=0), fn.std(axis=0)

        # Step 4: AUT on the seed-averaged PR-AUC curve (not average of per-seed AUTs)
        aut_ben = compute_aut(pb_mean.tolist())
        aut_att = compute_aut(pa_mean.tolist())

        # Step 5: print grid table with per-task mean ± std and AUT in last column
        print(f"\n{'='*80}")
        print(f"  Per-task results ({split_label}) — mean ± std across {len(seed_list)} seeds")
        print(f"{'='*80}")
        header = ["Metric"] + [f"T{i}" for i in range(n_tasks)] + ["AUT"]
        rows = [
            ["PR-AUC Benign (mean)"] + [f"{v:.4f}" for v in pb_mean] + [f"{aut_ben:.4f}"],
            ["PR-AUC Benign (std) "] + [f"{v:.4f}" for v in pb_std]  + ["---"],
            ["PR-AUC Attack (mean)"] + [f"{v:.4f}" for v in pa_mean] + [f"{aut_att:.4f}"],
            ["PR-AUC Attack (std) "] + [f"{v:.4f}" for v in pa_std]  + ["---"],
            ["FPR (mean)          "] + [f"{v:.4f}" for v in fr_mean] + ["---"],
            ["FPR (std)           "] + [f"{v:.4f}" for v in fr_std]  + ["---"],
            ["FNR (mean)          "] + [f"{v:.4f}" for v in fn_mean] + ["---"],
            ["FNR (std)           "] + [f"{v:.4f}" for v in fn_std]  + ["---"],
        ]
        print(tabulate(rows, headers=header, tablefmt="grid"))

        # Step 6: per-seed raw values (not just the mean±std above) -- one
        # block of rows per seed, same metrics/tasks, no aggregation.
        print(f"\n  Per-seed results ({split_label}) — raw values, {len(seed_list)} seeds")
        seed_ids = list(auc_results.keys())
        seed_rows = []
        for si, seed_id in enumerate(seed_ids):
            seed_rows.append([f"seed={seed_id} PR-AUC Benign"] + [f"{v:.4f}" for v in pb[si]])
            seed_rows.append([f"seed={seed_id} PR-AUC Attack"] + [f"{v:.4f}" for v in pa[si]])
            seed_rows.append([f"seed={seed_id} FPR"]           + [f"{v:.4f}" for v in fr[si]])
            seed_rows.append([f"seed={seed_id} FNR"]           + [f"{v:.4f}" for v in fn[si]])
        print(tabulate(seed_rows, headers=["seed / metric"] + [f"T{i}" for i in range(n_tasks)], tablefmt="grid"))

    # Print results for each split
    _per_task_stats(result_key=0, split_label="seen tasks")
    _per_task_stats(result_key=1, split_label="unseen tasks")
    _per_task_stats(result_key=6, split_label="all tasks")

    # ── Self-label (OWL) accuracy: per-task, per-class, averaged across seeds ──
    # result_key=7 = SELF_LABEL_ACCURACY_LOG, a list of per-unseen-task dicts:
    #   {'task_id', 'class0_correct','class0_total','class1_correct','class1_total'}
    # accuracy = correct/total against ground truth, on the samples the OWL
    # mechanism actually self-labeled and trained on for that task.
    def _self_label_accuracy_stats():
        # per_task[task_id] -> list of (class0_acc, class1_acc, combined_acc) across seeds
        per_task = {}
        # per_task_n[task_id] -> list of (class0_total, class1_total, combined_total) across seeds
        per_task_n = {}
        # for the single cross-task metric: pool raw counts across seeds+tasks (micro),
        # and separately track each seed's own macro-avg-over-tasks (for macro std)
        micro_correct, micro_total = 0, 0
        seed_macro_accs = []
        for seed_res in auc_results.values():
            log = seed_res[7] if len(seed_res) > 7 else []
            this_seed_task_accs = []
            for rec in log:
                tid = rec['task_id']
                c0c, c0t = rec['class0_correct'], rec['class0_total']
                c1c, c1t = rec['class1_correct'], rec['class1_total']
                c0_acc = c0c / c0t if c0t > 0 else float('nan')
                c1_acc = c1c / c1t if c1t > 0 else float('nan')
                t_total = c0t + c1t
                t_acc = (c0c + c1c) / t_total if t_total > 0 else float('nan')
                per_task.setdefault(tid, []).append((c0_acc, c1_acc, t_acc))
                per_task_n.setdefault(tid, []).append((c0t, c1t, t_total))
                if t_total > 0:
                    this_seed_task_accs.append(t_acc)
                micro_correct += (c0c + c1c)
                micro_total += t_total
            if this_seed_task_accs:
                seed_macro_accs.append(np.nanmean(this_seed_task_accs))

        if not per_task:
            print("\n  Skipping self-label accuracy: no SELF_LABEL_ACCURACY_LOG data found "
                  "(older run predating this metric?)")
            return

        print(f"\n{'='*80}")
        print(f"  Self-label (OWL) accuracy per task — mean ± std across {len(seed_list)} seeds")
        print(f"{'='*80}")
        rows = []
        for tid in sorted(per_task.keys()):
            vals = np.array(per_task[tid], dtype=float)  # shape (n_seeds, 3): c0,c1,combined
            means = np.nanmean(vals, axis=0) * 100
            stds = np.nanstd(vals, axis=0) * 100
            n_vals = np.array(per_task_n[tid], dtype=float)
            n_means = n_vals.mean(axis=0)
            rows.append([f"T{tid}",
                         f"{means[0]:.2f} ± {stds[0]:.2f}",
                         f"{n_means[0]:.1f}",
                         f"{means[1]:.2f} ± {stds[1]:.2f}",
                         f"{n_means[1]:.1f}",
                         f"{means[2]:.2f} ± {stds[2]:.2f}",
                         f"{n_means[2]:.1f}"])
        print(tabulate(rows, headers=["task", "class0 (benign) % acc", "class0 n (mean)",
                                       "class1 (attack) % acc", "class1 n (mean)",
                                       "combined % acc", "combined n (mean)"],
                        tablefmt="grid"))

        # Single metric across all tasks:
        #  - macro: mean of each seed's own (per-task-averaged) combined accuracy,
        #    then mean/std of that across seeds -- every task weighted equally.
        #  - micro: pooled correct/total across every task and every seed -- every
        #    self-labeled sample weighted equally (dominated by high-volume tasks).
        macro_mean = np.nanmean(seed_macro_accs) * 100 if seed_macro_accs else float('nan')
        macro_std = np.nanstd(seed_macro_accs) * 100 if seed_macro_accs else float('nan')
        micro_acc = (micro_correct / micro_total * 100) if micro_total > 0 else float('nan')
        print(f"\n  Self-label accuracy — single metric across all tasks:")
        print(f"    macro (mean over tasks, then over {len(seed_macro_accs)} seeds) = {macro_mean:.2f}% ± {macro_std:.2f}%")
        print(f"    micro (sample-weighted, pooled over all tasks & seeds)          = {micro_acc:.2f}%  ({micro_correct}/{micro_total})")

    _self_label_accuracy_stats()

    print("-"*80)
    total_time = time.time()-start_time
    print("total execution time is %.3f seconds" % (total_time))
    print("avg execution time %.3f seconds"%(total_time/len(seed_list)))