import subprocess
import os
import tempfile
import json
import argparse
import time
from tabulate import tabulate
import numpy as np





if __name__ == "__main__":
    start_time=time.time()
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)') 
    parser.add_argument('--ds', type=str, default="anoshift", metavar='S',help='dataset name')
    parser.add_argument('--b_m', type=float, default=0.2, metavar='S',help='batch memory ratio(default: 0.2)')
    # parser.add_argument('--lr', type=float, default=1e-4, metavar='S',help='batch memory ratio(default: 0.001)')
    # parser.add_argument('--wd', type=float, default=1e-5, metavar='S',help='batch memory ratio(default: 0.01)')
    # parser.add_argument('--lr', type=float, default=1e-4, metavar='S',help='batch memory ratio(default: 0.001)')
    # parser.add_argument('--wd', type=float, default=1e-5, metavar='S',help='batch memory ratio(default: 0.01)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='S',help='batch memory ratio(default: 0.001)')
    parser.add_argument('--wd', type=float, default=1e-5, metavar='S',help='batch memory ratio(default: 0.01)')
    parser.add_argument('--label_ratio', type=float, default=0.2, metavar='S',help='labeled ratio (default: 0.1)')
    parser.add_argument('--nps', type=int, metavar='S',default=10000,help='number of projection samples(default: 100)')
    parser.add_argument('--bma', type=float, metavar='S',default=0.2,help='batch minority allocation(default: 0)')
    parser.add_argument('--alpha', type=float, metavar='S',default=1,help='distill loss multiplier(default: 0)')
    parser.add_argument('--lab_samp_in_mem_ratio', type=float, metavar='S',default=1.0,help='Percentage of labeled samples to store in memory(default: 1.0)')
    parser.add_argument('--bool_gpm', type=str, metavar='S',default="True",help='Enables gradient projections(default: True)')
    parser.add_argument('--mem_strat', type=str, metavar='S',default="equal",help='Buffer memory strategy(default: full initialization)')
    parser.add_argument('--training_cutoff', type=int, default=3, metavar='S',help='train the model for first n tasks and test for time decay on the rest')
    parser.add_argument('--bool_closs', type=str, metavar='S',default="False",help='Enables using contrastive loss(default: False)')
    parser.add_argument('--mlps', type=int, metavar='S',default=1,help='Number of learners (MLPs)default: 1)')
    parser.add_argument('--batch_size', type=int, metavar='S',default=1024,help='batch size: 1024)')
    parser.add_argument('--train_with_unlab', type=str, metavar='S',default="True",help='Sets to train with unlabeled data(default: True)')
    parser.add_argument('--cos_dist', type=float, metavar='S',default=0.05,help='cosine distance for OWL(default: 0.1)')
    parser.add_argument('--mode_val', type=int, metavar='S',default=98,help='Mode value for OWL (default: 98)')

    args = parser.parse_args()
    auc_results = {}
    seed_list = [1,2,3]
    # seed_list = [2]
    curr_dir = os.getcwd()
    for seed_value in seed_list:
        print("seed is",seed_value)
        fd, temp_file_name = tempfile.mkstemp() # create temporary file
        
        os.close(fd) # close the file
        proc = subprocess.Popen(["python anoshift_spider_owl.py --seed="+str(seed_value)+" --ds="+str(args.ds)+" --lr="+str(args.lr)+" --wd="+str(args.wd)+" --batch_size="+str(args.batch_size)+" --alpha="+str(args.alpha)+" --gpu="+str(args.gpu)+" --filename="+str(temp_file_name)+" --cos_dist="+str(args.cos_dist)+" --mode_val="+str(args.mode_val)+" --train_with_unlab="+str(args.train_with_unlab)+" --bool_gpm="+str(args.bool_gpm)+" --mem_strat="+str(args.mem_strat)+" --b_m="+str(args.b_m)+" --label_ratio="+str(args.label_ratio)+" --lab_samp_in_mem_ratio="+str(args.lab_samp_in_mem_ratio)+" --nps="+str(args.nps)+" --bma="+str(args.bma)+" --training_cutoff="+str(args.training_cutoff)+" --bool_closs="+str(args.bool_closs)+" --mlps="+str(args.mlps)],shell=True,cwd=curr_dir)
        # proc = subprocess.Popen(["python anoshift_spider_owl_shreya.py --seed="+str(seed_value)+" --ds="+str(args.ds)+" --batch_size="+str(args.batch_size)+" --lr="+str(args.lr)+" --wd="+str(args.wd)+" --alpha="+str(args.alpha)+" --gpu="+str(args.gpu)+" --filename="+str(temp_file_name)+" --bool_gpm="+str(args.bool_gpm)+" --mem_strat="+str(args.mem_strat)+" --b_m="+str(args.b_m)+" --label_ratio="+str(args.label_ratio)+" --lab_samp_in_mem_ratio="+str(args.lab_samp_in_mem_ratio)+" --nps="+str(args.nps)+" --bma="+str(args.bma)+" --training_cutoff="+str(args.training_cutoff)+" --bool_closs="+str(args.bool_closs)+" --mlps="+str(args.mlps)],shell=True,cwd=curr_dir)
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

    # Print results for each split
    _per_task_stats(result_key=0, split_label="seen tasks")
    _per_task_stats(result_key=1, split_label="unseen tasks")
    _per_task_stats(result_key=6, split_label="all tasks")

    print("-"*80)
    total_time = time.time()-start_time
    print("total execution time is %.3f seconds" % (total_time))
    print("avg execution time %.3f seconds"%(total_time/len(seed_list)))