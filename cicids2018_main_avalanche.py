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
    parser.add_argument('--ds', type=str, default="ids18", metavar='S',help='dataset name')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='S',help='batch memory ratio(default: 0.2)')
    parser.add_argument('--w_d', type=float, default=1e-4, metavar='S',help='labeled ratio (default: 0.1)')
    parser.add_argument('--cl_strat', type=int, default=0, metavar='S',help='integer id associated with the CL strategy)')
    parser.add_argument('--training_cutoff', type=int, default=5, metavar='S',help='train the model for first n tasks and test for time decay on the rest')



    args = parser.parse_args()
    auc_results = {}
    seed_list = [1,2,3]
    curr_dir = os.getcwd()
    for seed_value in seed_list:
        print("seed is",seed_value)
        fd, temp_file_name = tempfile.mkstemp() # create temporary file
        
        os.close(fd) # close the file
        proc = subprocess.Popen(["python cicids2018_avalanche.py --seed="+str(seed_value)+" --ds="+str(args.ds)+" --gpu="+str(args.gpu)+" --training_cutoff="+str(args.training_cutoff)+" --filename="+str(temp_file_name)+" --w_d="+str(args.w_d)+' --cl_strat='+str(args.cl_strat)+" --lr="+str(args.lr)],shell=True,cwd=curr_dir)
        proc.communicate()
        with open(temp_file_name) as fp:
            result = json.load(fp)
            # auc_results[str(seed_value)] = result#result[str(seed_value)]
            auc_results[str(seed_value)] = result[str(seed_value)]
        os.unlink(temp_file_name)    

    # print("{:<20}  {:<20}".format('Argument','Value'))
    # print("*"*80)
    # for arg in vars(args):
    #     print("{:<20}  {:<20}".format(arg, getattr(args, arg)))
    # print("*"*80)    
    # print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format('seed','PR-AUC(O)', 'PR-AUC(I)', 'ROC-AUC','Total train time'))
    # print("*"*80)
    print("{:<20}  {:<20}".format('Argument','Value'))
    print("*"*80)
    for arg in vars(args):
        print("{:<20}  {:<20}".format(arg, getattr(args, arg)))
    print("*"*80)
    print("*"*80)

    # ── Helper: AUT via trapezoidal rule over per-task PR-AUC values ─────────────
    # AUT = (1/(N-1)) * sum of trapezoids between consecutive tasks.
    # This is computed on the seed-averaged curve, not averaged over per-seed AUTs.
    def compute_aut(prauc_list):
        n = len(prauc_list)
        if n < 2:
            return float('nan')
        return sum((prauc_list[i] + prauc_list[i + 1]) / 2 for i in range(n - 1)) / (n - 1)

    # ── Per-task PR-AUC, FPR, FNR averaged across seeds + AUT on averaged curve ──
    # result_key mapping (no OWL, so all-tasks is at index 2, not 6):
    #   0 = seen tasks result   (tasks 0 .. training_cutoff-1)
    #   1 = unseen tasks result (tasks training_cutoff .. end)
    #   2 = all tasks result    (tasks 0 .. end)
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
    _per_task_stats(result_key=2, split_label="all tasks")

    print("-"*80)
    total_time = time.time()-start_time
    print("\ntotal execution time is %.3f seconds" % (total_time))
    print("avg execution time %.3f seconds"%(total_time/len(seed_list)))
