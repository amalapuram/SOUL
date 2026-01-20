import subprocess
import os
import tempfile
import json
import argparse
import time
from tabulate import tabulate



if __name__ == "__main__":
    start_time=time.time()
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)') 
    parser.add_argument('--ds', type=str, default="apigraph", metavar='S',help='dataset name')
    parser.add_argument('--b_m', type=float, default=0, metavar='S',help='batch memory ratio(default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='S',help='batch memory ratio(default: 0.001)')
    parser.add_argument('--wd', type=float, default=1e-6, metavar='S',help='batch memory ratio(default: 0.01)')
    parser.add_argument('--label_ratio', type=float, default=0.2, metavar='S',help='labeled ratio (default: 0.1)')
    parser.add_argument('--nps', type=int, metavar='S',default=10000,help='number of projection samples(default: 100)')
    parser.add_argument('--bma', type=float, metavar='S',default=0.8,help='batch minority allocation(default: 0)')
    parser.add_argument('--alpha', type=float, metavar='S',default=9,help='distill loss multiplier(default: 0)')
    parser.add_argument('--lab_samp_in_mem_ratio', type=float, metavar='S',default=0.1,help='Percentage of labeled samples to store in memory(default: 1.0)')
    parser.add_argument('--bool_gpm', type=str, metavar='S',default="True",help='Enables gradient projections(default: True)')
    parser.add_argument('--mem_strat', type=str, metavar='S',default="equal",help='Buffer memory strategy(default: full initialization)')
    parser.add_argument('--training_cutoff', type=int, default=4, metavar='S',help='train the model for first n tasks and test for time decay on the rest')
    parser.add_argument('--bool_closs', type=str, metavar='S',default="False",help='Enables using contrastive loss(default: False)')
    parser.add_argument('--mlps', type=int, metavar='S',default=1,help='Number of learners (MLPs)default: 1)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch_size')


    args = parser.parse_args()
    auc_results = {}
    seed_list = [1, 2, 4]
    curr_dir = os.getcwd()
    for seed_value in seed_list:
        print("seed is",seed_value)
        fd, temp_file_name = tempfile.mkstemp() # create temporary file
        
        os.close(fd) # close the file
        proc = subprocess.Popen(["python apigraph_spider_owl.py --seed="+str(seed_value)+" --ds="+str(args.ds)+" --lr="+str(args.lr)+" --wd="+str(args.wd)+" --batch_size="+str(args.batch_size)+" --alpha="+str(args.alpha)+" --gpu="+str(args.gpu)+" --filename="+str(temp_file_name)+" --bool_gpm="+str(args.bool_gpm)+" --mem_strat="+str(args.mem_strat)+" --b_m="+str(args.b_m)+" --label_ratio="+str(args.label_ratio)+" --lab_samp_in_mem_ratio="+str(args.lab_samp_in_mem_ratio)+" --nps="+str(args.nps)+" --bma="+str(args.bma)+" --training_cutoff="+str(args.training_cutoff)+" --bool_closs="+str(args.bool_closs)+" --mlps="+str(args.mlps)],shell=True,cwd=curr_dir)
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
    aut_results = {}
    for key, value in auc_results.items():
        print("training results for seed value",key)
        prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N = value[0][0],value[0][1],value[0][2],value[0][3],value[0][4],value[0][5],value[0][6]
        pnt_table = [
        # ['task_CI']+ task_CI_pnt, 
        # ['test_CI'] + test_CI_pnt,
        ['prauc Benign traffic'] + prauc_in_pnt, 
        ['prauc Attack traffic'] + prauc_out_pnt
    ]
        print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
        print(f'AUT(prauc inliers,{N}) := {prauc_in_aut}')
        print(f'AUT(prauc outliers,{N}) := {prauc_out_aut}')
        print("testing results for seed value",key)
        prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N = value[1][0],value[1][1],value[1][2],value[1][3],value[1][4],value[1][5],value[1][6]
        pnt_table = [ # ['task_CI']+ task_CI_pnt, 
                # ['test_CI'] + test_CI_pnt,
                ['prauc Benign traffic'] + prauc_in_pnt, 
                ['prauc Attack traffic'] + prauc_out_pnt
                ]
        print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
        print(f'AUT(prauc inliers,{N}) := {prauc_in_aut}')
        print(f'AUT(prauc outliers,{N}) := {prauc_out_aut}')
        aut_results[key] = [prauc_in_aut,prauc_out_aut,float(value[2]),float(value[3]),float(value[4]),float(value[5])]

    print("-"*80)
    
    aut_results_values = list(aut_results.values())
    aut_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*aut_results_values)]
    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format('Cols','AUT(Benign)','AUT(Attack)','Self_labels (Benign)', 'Self_labels (Attack)','Analyst_labels (Benign)', 'Analyst_labels (Attack)'))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format('Averge',float(str(aut_average[0])[:5]), float(str(aut_average[1])[:5]),float(str(aut_average[2])),float(str(aut_average[3])),float(str(aut_average[4])),float(str(aut_average[5]))))
    print("-"*80)
    total_time = time.time()-start_time
    print("total execution time is %.3f seconds" % (total_time))
    print("avg execution time %.3f seconds"%(total_time/len(seed_list)))