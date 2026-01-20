import subprocess
import os
import tempfile
import json
import argparse
import time
import numpy as np
from tabulate import tabulate






if __name__ == "__main__":
    start_time=time.time()
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)') 
    parser.add_argument('--ds', type=str, default="anoshift", metavar='S',help='dataset name')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='S',help='batch memory ratio(default: 0.2)')
    parser.add_argument('--w_d', type=float, default=1e-5, metavar='S',help='labeled ratio (default: 0.1)')
    # parser.add_argument('--wd', type=float, default= 1e-05, metavar='S',help='optim weight decay')
    parser.add_argument('--training_cutoff', type=int, default=3, metavar='S',help='train the model for first n tasks and test for time decay on the rest')


    args = parser.parse_args()
    auc_results = {}
    seed_list = [1,2,3]
    
    curr_dir = os.getcwd()
    for seed_value in seed_list:
        print("seed is",seed_value)
        fd, temp_file_name = tempfile.mkstemp() # create temporary file
        
        os.close(fd) # close the file
        proc = subprocess.Popen(["python anoshift_MIR.py --seed="+str(seed_value)+" --lr="+str(args.lr)+" --w_d="+str(args.w_d)+" --ds="+str(args.ds)+" --gpu="+str(args.gpu)+" --training_cutoff="+str(args.training_cutoff)+" --filename="+str(temp_file_name)],shell=True,cwd=curr_dir)
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
    # print("{:<20}  {:<20}  {:<20}  {:<20}  {:<20}  {:<20} {:<20}  {:<20}".format('seed','PR-AUC(O)', 'PR-AUC(I)', 'ROC-AUC','Total train time','Total MIR time','Regular SGD updates','Virtual SGD updates'))
    # print("*"*80)
        
    
    aut_results = {}
    for key, value in auc_results.items():
        # print("training results for seed value",key)
        prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N = value[0][0],value[0][1],value[0][2],value[0][3],value[0][4],value[0][5],value[0][6]
        aut_results[key] = [prauc_in_aut,prauc_out_aut]
    #     pnt_table = [
    #     # ['task_CI']+ task_CI_pnt, 
    #     # ['test_CI'] + test_CI_pnt,
    #     ['prauc Benign traffic'] + prauc_in_pnt, 
    #     ['prauc Attack traffic'] + prauc_out_pnt
    # ]
    #     print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
    #     print(f'AUT(prauc inliers,{N}) := {prauc_in_aut}')
    #     print(f'AUT(prauc outliers,{N}) := {prauc_out_aut}')
    #     print("testing results for seed value",key)
        prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N = value[1][0],value[1][1],value[1][2],value[1][3],value[1][4],value[1][5],value[1][6]
        # pnt_table = [ # ['task_CI']+ task_CI_pnt, 
        #         # ['test_CI'] + test_CI_pnt,
        #         ['prauc Benign traffic'] + prauc_in_pnt, 
        #         ['prauc Attack traffic'] + prauc_out_pnt
        #         ]
        # print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
        # print(f'AUT(prauc inliers,{N}) := {prauc_in_aut}')
        # print(f'AUT(prauc outliers,{N}) := {prauc_out_aut}')
        aut_results[key].extend([prauc_in_aut,prauc_out_aut])
        prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N = value[2][0],value[2][1],value[2][2],value[2][3],value[2][4],value[2][5],value[2][6]
        aut_results[key].extend([prauc_in_aut,prauc_out_aut])

    print("-"*80)
    
    
    aut_results_values = list(aut_results.values())
    
    # aut_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*aut_results_values)]
    aut_average = (np.mean(np.array(list(aut_results.values())),axis=0)).tolist()
    aut_std = (np.std(np.array(list(aut_results.values())),axis=0)).tolist()
    print("{:<20}  {:<20}  {:<20}  ".format('Cols','AUT(Benign)-seen','AUT(Attack)-seen'))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20} ".format('Mean',float(str(aut_average[0])[:5]), float(str(aut_average[1]))))    
    print("{:<20}  {:<20}  {:<20} ".format('std',float(str(aut_std[0])[:5]), float(str(aut_std[1]))))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20}  ".format('Cols','AUT(Benign)-unseen','AUT(Attack)-unseen'))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20} ".format('Mean',float(str(aut_average[2])[:5]), float(str(aut_average[3]))))    
    print("{:<20}  {:<20}  {:<20} ".format('std',float(str(aut_std[2])[:5]), float(str(aut_std[3]))))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20}  ".format('Cols','AUT(Benign)-all','AUT(Attack)-all'))
    print("-"*80)
    print("{:<20}  {:<20}  {:<20} ".format('Mean',float(str(aut_average[4])[:5]), float(str(aut_average[5]))))    
    print("{:<20}  {:<20}  {:<20} ".format('std',float(str(aut_std[4])[:5]), float(str(aut_std[5]))))
    total_time = time.time()-start_time
    print("\ntotal execution time is %.3f seconds" % (total_time))
    print("avg execution time %.3f seconds"%(total_time/len(seed_list)))
