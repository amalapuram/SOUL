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
    parser.add_argument('--ds', type=str, default="androzoo", metavar='S',help='dataset name')
    parser.add_argument('--b_m', type=float, default=0.3, metavar='S',help='batch memory ratio(default: 0.2)')
    # parser.add_argument('--lr', type=float, default=1e-4, metavar='S',help='batch memory ratio(default: 0.001)')
    # parser.add_argument('--wd', type=float, default=1e-5, metavar='S',help='batch memory ratio(default: 0.01)')
    parser.add_argument('--label_ratio', type=float, default=0.1, metavar='S',help='labeled ratio (default: 0.1)')
    parser.add_argument('--nps', type=int, metavar='S',default=10000,help='number of projection samples(default: 100)')
    parser.add_argument('--bma', type=float, metavar='S',default=0.8,help='batch minority allocation(default: 0)')
    parser.add_argument('--alpha', type=float, metavar='S',default=9,help='distill loss multiplier(default: 0)')
    parser.add_argument('--lab_samp_in_mem_ratio', type=float, metavar='S',default=1.0,help='Percentage of labeled samples to store in memory(default: 1.0)')
    parser.add_argument('--bool_gpm', type=str, metavar='S',default="True",help='Enables gradient projections(default: True)')
    parser.add_argument('--mem_strat', type=str, metavar='S',default="equal",help='Buffer memory strategy(default: full initialization)')
    parser.add_argument('--training_cutoff', type=int, default=3, metavar='S',help='train the model for first n tasks and test for time decay on the rest')
    parser.add_argument('--bool_closs', type=str, metavar='S',default="False",help='Enables using contrastive loss(default: False)')
    parser.add_argument('--mlps', type=int, metavar='S',default=1,help='Number of learners (MLPs)default: 1)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N', help='number of training epochs/task (default: 10)')
    
    args = parser.parse_args()
    auc_results = {}
    curr_dir = os.getcwd() 

    lr_list = [1e-2, 1e-3, 1e-4, 1e-5,1e-6,1e-7,1e-8] 
    wd_list = [1e-1,1e-2,1e-3, 1e-4, 1e-5,1e-6,1e-7]
    batch_sizes = [16, 32, 64, 128, 256]

    best_aut = 0.0
    best_lr, best_wd, best_batch_size = None, None, None


    for i, lr in enumerate(lr_list):
        for j, wd in enumerate(wd_list):
            for k, batch_size in enumerate(batch_sizes):
                print(f'Learning rate = {lr}, Weight decay = {wd}, batch_size = {batch_size}')

                fd, temp_file_name = tempfile.mkstemp() # create temporary file
                os.close(fd) # close the file
                proc = subprocess.Popen(["python androzoo_spider_owl.py --grid_search=True"+" --ds="+str(args.ds)+" --lr="+str(lr)+" --wd="+str(wd)+" --batch_size="+str(batch_size)+" --alpha="+str(args.alpha)+" --gpu="+str(args.gpu)+" --filename="+str(temp_file_name)+" --bool_gpm="+str(args.bool_gpm)+" --mem_strat="+str(args.mem_strat)+" --b_m="+str(args.b_m)+" --label_ratio="+str(args.label_ratio)+" --lab_samp_in_mem_ratio="+str(args.lab_samp_in_mem_ratio)+" --nps="+str(args.nps)+" --bma="+str(args.bma)+" --training_cutoff="+str(args.training_cutoff)+" --bool_closs="+str(args.bool_closs)+" --mlps="+str(args.mlps)+" --n_epochs="+str(args.n_epochs)],shell=True,cwd=curr_dir)
                proc.communicate()
                with open(temp_file_name) as fp:
                    result = json.load(fp)
                    curr_aut = result['validation_aut_seen_minority']

                os.unlink(temp_file_name) 

                # aut_values[i][j] = curr_aut
                if curr_aut > best_aut:
                    best_aut = curr_aut
                    best_lr = lr
                    best_wd = wd
                    best_batch_size = batch_size

    print('For androzoo dataset in owl setting: (attack + benign shift)')
    print(f'Best lr = {best_lr}')
    print(f'Best wd = {best_wd}')
    print(f'Best batch_size = {best_batch_size}')
    print(f'Best aut value = {best_aut}')

    total_time = time.time()-start_time
    print("\ntotal execution time is %.3f seconds" % (total_time))

    # print(f'\nPrinting the grid: (rows - lr), (columns - wd): \n{aut_values}')