import subprocess
import os
import tempfile
import json
import argparse
import time





if __name__ == "__main__":
    start_time=time.time()
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)') 
    parser.add_argument('--ds', type=str, default="ids18", metavar='S',help='dataset name')
    parser.add_argument('--b_m', type=float, default=0.2, metavar='S',help='batch memory ratio(default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='S',help='batch memory ratio(default: 0.001)')
    parser.add_argument('--wd', type=float, default=0.01, metavar='S',help='batch memory ratio(default: 0.01)')
    parser.add_argument('--label_ratio', type=float, default=0.05, metavar='S',help='labeled ratio (default: 0.1)')
    parser.add_argument('--nps', type=int, metavar='S',default=100,help='number of projection samples(default: 100)')
    parser.add_argument('--bma', type=float, metavar='S',default=0.8,help='batch minority allocation(default: 0)')
    parser.add_argument('--alpha', type=float, metavar='S',default=9,help='distill loss multiplier(default: 0)')
    parser.add_argument('--lab_samp_in_mem_ratio', type=float, metavar='S',default=1.0,help='Percentage of labeled samples to store in memory(default: 1.0)')
    parser.add_argument('--bool_gpm', type=str, metavar='S',default="True",help='Enables gradient projections(default: True)')
    parser.add_argument('--mem_strat', type=str, metavar='S',default="equal",help='Buffer memory strategy(default: full initialization)')

    args = parser.parse_args()
    auc_results = {}
    seed_list = [1,2,3,4,5]
    curr_dir = os.getcwd()
    for seed_value in seed_list:
        print("seed is",seed_value)
        fd, temp_file_name = tempfile.mkstemp() # create temporary file
        
        os.close(fd) # close the file
        proc = subprocess.Popen(["python cicids2018_spider.py --seed="+str(seed_value)+" --ds="+str(args.ds)+" --lr="+str(args.lr)+" --wd="+str(args.wd)+" --alpha="+str(args.alpha)+" --gpu="+str(args.gpu)+" --filename="+str(temp_file_name)+" --bool_gpm="+str(args.bool_gpm)+" --mem_strat="+str(args.mem_strat)+" --b_m="+str(args.b_m)+" --label_ratio="+str(args.label_ratio)+" --lab_samp_in_mem_ratio="+str(args.lab_samp_in_mem_ratio)+" --nps="+str(args.nps)+" --bma="+str(args.bma)],shell=True,cwd=curr_dir)
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
    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20} {:<10}".format('seed','PR-AUC(O)', 'PR-AUC(I)', 'ROC-AUC','grad_norm_mean','grad_norm_variance'))
    print("*"*80)
    for key, value in auc_results.items():
        # print(key,value)
        pr_auc_o, pr_auc_1,roc_auc,grad_norm_mean,grad_norm_var = value
        print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format(key,pr_auc_o, pr_auc_1,roc_auc,grad_norm_mean,grad_norm_var))
    print("-"*80)
    auc_results_values = list(auc_results.values())
    auc_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*auc_results_values)]
    print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20}".format('avg',float(str(auc_average[0])[:5]), float (str(auc_average[1])[:5]), float(str(auc_average[2])[:5]), float (str(auc_average[3])[:5]),float(str(auc_average[4])[:5])))
    print("-"*80)
    total_time = time.time()-start_time
    print("total execution time is %.3f seconds" % (total_time))
    print("avg execution time %.3f seconds"%(total_time/len(seed_list)))