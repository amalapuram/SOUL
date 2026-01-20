from turtle import st
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torch.optim.lr_scheduler import StepLR
# from pytorchtools import EarlyStopping
import subprocess
import os
import tempfile
import numpy as np
import pandas as pd
import statistics

from utils.customdataloader import load_dataset,Tempdataset,compute_total_minority_testsamples,get_inputshape,load_teset
from utils.buffermemory import memory_update,retrieve_replaysamples,memory_update_equal_allocation,memory_update_equal_allocation2,memory_update_equal_allocation3
from utils.metrics import compute_results,plot_tsne,plot_grad_norm_line_graph
from utils.utils import log,create_directories,trigger_logging,set_seed,get_gpu,load_model,get_dataset_info,EarlyStopping,GradientRejection
from utils.config.configurations import cfg
# from utils.config.configurations import cifar100 as ds
from utils.metadata import initialize_metadata
from utils.config.custom_resnet import custom_resnet18
from utils.config.custom_reduced_resnet import reduced_resnet18
from utils.config.custom_alexnet import AlexNet



import time
import random
from math import floor
from collections import Counter
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
from tqdm import tqdm
import itertools
import argparse
import json



from torchmetrics import Accuracy
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

import warnings
warnings.filterwarnings("ignore")

scaler = MinMaxScaler()

memory_population_time=0
global_priority_list=dict()
local_priority_list=dict()
local_count = Counter()
classes_so_far = set()
full = set()
local_store = {}
global_count, local_count, replay_count,replay_individual_count = Counter(), Counter(),Counter(),Counter()
input_shape,task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,learning_rate,ecbrs_taskaware = None,None,None,None,None,None,None,None,None,None
replay_size,memory_size,minority_allocation,epochs,batch_size,device,pattern_per_exp,is_lazy_training,task_num = None,None,None,None,None,None,None,None,None
memory_X, memory_y, memory_y_name,ecbrs_taskaware_memory_X, ecbrs_taskaware_memory_y,ecbrs_taskaware_memory_y_name,memory_per_task = None,None,None,None,None,None,None
model,teacher_model,opt,loss_fn,train_acc_metric = None,None,None,None,None
pth_testset,testset_class_ids =None,None
test_x,test_y = None,None
image_resolution = None
bool_encode_anomaly,bool_encode_benign,load_whole_train_data=None,None,None
nc = 0
no_tasks = 0
grad_norm_dict = []
temp_norm =1
# labels_ratio, no_of_rand_samples, minority_alloc = None,None,None





def load_metadata(dataset_name,l_rate,w_decay):
   
    global task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,learning_rate,input_shape,ecbrs_taskaware,image_resolution,pth_testset,testset_class_ids,bool_encode_anomaly,bool_encode_benign,load_whole_train_data
    global replay_size,memory_size,minority_allocation,epochs,batch_size,device,pattern_per_exp,is_lazy_training,ecbrs_taskaware_memory_X, ecbrs_taskaware_memory_y,ecbrs_taskaware_memory_y_name,memory_per_task
    
    ds = get_dataset_info(dataset_name)
    label = ds.label
    cfg.avalanche_dir = False
    set_cl_strategy_name(1)
    no_tasks = ds.no_tasks
    metadata_dict = initialize_metadata(label)
    temp_dict = metadata_dict[no_tasks]
    task_order = temp_dict['task_order']
    class_ids = temp_dict['class_ids']
    minorityclass_ids = temp_dict['minorityclass_ids']
    pth = temp_dict['path']
    if 'path_testset' in temp_dict:
        pth_testset = temp_dict['path_testset']
        testset_class_ids = temp_dict['testset_class_ids']
    tasks_list = temp_dict['tasks_list']
    task2_list = temp_dict['task2_list']
    replay_size = ds.replay_size
    memory_size = ds.mem_size
    # memory_size = ds.mem_size_semi_supervised
    minority_allocation = ds.minority_allocation
    epochs = 100#ds.n_epochs
    batch_size = ds.batch_size
    device = cfg.device
    learning_rate = l_rate# 0.001#ds.learning_rate
    no_tasks = ds.no_tasks
    pattern_per_exp = ds.pattern_per_exp
    is_lazy_training = ds.is_lazy_training
    ecbrs_taskaware = ds.taskaware_ecbrs
    bool_encode_anomaly = ds.bool_encode_anomaly
    bool_encode_benign = ds.bool_encode_benign
    load_whole_train_data = ds.load_whole_train_data
    if ecbrs_taskaware:
        minority_allocation = ds.taskaware_minority_allocation
    input_shape = get_inputshape(pth,class_ids)
    compute_total_minority_testsamples(pth=pth,dataset_label=label,minorityclass_ids=minorityclass_ids,no_tasks=no_tasks)
    load_model_metadata(w_decay)
    create_directories(label)
    trigger_logging(label=label)
    image_resolution = ds.image_resolution
    if ecbrs_taskaware:
        set_cl_strategy_name(2)
        ecbrs_taskaware_memory_X = np.zeros((1,input_shape))
        ecbrs_taskaware_memory_y = np.zeros(1)
        ecbrs_taskaware_memory_y_name = np.zeros(1)
        memory_per_task = ds.pattern_per_exp
        memory_size = memory_per_task

def load_model_metadata(w_decay):
    log("loading model parameter")
    global model,opt,loss_fn,train_acc_metric,input_shape,teacher_model
    model = load_model(label=label,inputsize=get_inputshape(pth,class_ids))
    # print(model)
    teacher_model = load_model(label=label,inputsize=get_inputshape(pth,class_ids))
    model = model.to(device)
    teacher_model = teacher_model.to(device)
    # model.train()
    # opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    w_d = w_decay#0.01
    print("weight decay is",w_d)
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=w_d)
    # opt = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=0.0001)
    # opt = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=0.01)
    loss_fn = torch.nn.BCELoss()
    train_acc_metric = Accuracy().to(device)

def set_cl_strategy_name(strategy_id):
    if strategy_id == 0:
        cfg.clstrategy = "CBRS"            
    elif strategy_id == 1:
        cfg.clstrategy = "ECBRS"
    elif strategy_id == 2:
        cfg.clstrategy = "ECBRS_taskaware"    
     
          

def initialize_buffermemory(tasks,mem_size):
    global memory_X, memory_y, memory_y_name
    attack_class_indicies_list = []
    attack_class_indicies = []
    initial_X, initial_y, initial_yname = tasks[0]
    unique_class = [int(x) for x in np.unique(initial_yname)]
    attack_class = [int(x) for x in minorityclass_ids]
    # common_attack_class = [x for x in attack_class if x in unique_class]
    common_attack_class = unique_class
    # print("common attack classes",common_attack_class)
    # exit()
    for idx,class_idx in enumerate(common_attack_class):
         indices = list(np.where(initial_yname == int(class_idx))[0])
         attack_class_indicies_list.insert(idx,indices)


    for concat_list in itertools.zip_longest(*attack_class_indicies_list):
        attack_class_indicies.extend(list(concat_list))

    attack_class_indicies = [x for x in attack_class_indicies if x is not None]  
    if len(attack_class_indicies) > mem_size:
        attack_class_indicies = attack_class_indicies[0:mem_size]   

    memory_X, memory_y, memory_y_name = initial_X[attack_class_indicies,:], initial_y[attack_class_indicies], initial_yname[attack_class_indicies]
    







def update_mem_samples_indexdict(memorysamples):
    global local_store
    for idx,class_ in enumerate(memorysamples):
        if class_ in local_store :
            local_store[class_].append(idx)
        else:
            local_store[class_] = [idx]





def get_representation_matrix (net, device, x, y=None,rand_samples=1000): 
    # Collect activations by forward pass
    # benign_indices = random.sample(list(range(x.shape[0])),min(100000,x.shape[0]))
    # print(x.shape[0])
    
    
    print("number of rand samples",rand_samples)
    print(x.shape[0])
    
    benign_indices = (np.where(y == int(0))[0]).tolist()
    attack_indices = list(set(range(0,x.shape[0]))-set(benign_indices))
    a_min_length = min(len(attack_indices),rand_samples)
    b_min_length = min(len(benign_indices),rand_samples)
    print(a_min_length,b_min_length)
    random.shuffle(benign_indices)
    random.shuffle(attack_indices)
    benign_indices = benign_indices[0:b_min_length]
    benign_indices.extend(attack_indices[0:a_min_length])
    
                                    

    
    b=benign_indices
    rand_samples = len(benign_indices)
    x=torch.tensor(x,dtype=torch.float32).to(device)
    example_data = x[b]#.view(-1,70)
    example_data = example_data.to(device)
    example_out  = net(example_data)
    
    batch_list=[int(rand_samples),int(rand_samples),int(rand_samples),int(rand_samples),int(rand_samples),rand_samples,rand_samples] 
    mat_list=[] # list contains representation matrix of each layer
    act_key=list(net.act.keys())
    print("keys are",act_key)

    for i in range(len(act_key)):
        bsz=batch_list[i]
        act = net.act[act_key[i]].detach().cpu().numpy()
        activation = act[0:bsz].transpose()
        mat_list.append(activation)

    
    return mat_list


def update_GPM (model, mat_list, threshold, feature_list=[],):
    # print ('Threshold: ', threshold) 
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            feature_list.append(U[:,0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                # print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                continue
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]))  
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui
    
    return feature_list  




def split_a_task(task,lab_ratio,task_class_ids=None):
    global batch_size
    labeled_indices,unlabeled_indices = [],[]
    # print("task classes",task_class_ids)
    
    X,y,y_classname = task[0][0],task[0][1],task[0][2]
    # print("y class name",Counter(y_classname))
    for class_idx in np.unique(y_classname):
        indices = (np.where(y_classname == int(class_idx))[0]).tolist()
        # random.shuffle(indices)
        labeled_index = max(1,floor(len(indices)*lab_ratio))
        labeled_indices.extend(indices[0:labeled_index])
        unlabeled_indices.extend(indices[labeled_index:])
    random.shuffle(labeled_indices)    
    random.shuffle(unlabeled_indices)

    # print("****************************")
    # # labeled_indices.sort()
    # # print(labeled_indices)
    # print("labelled size",len(labeled_indices))
    # print("ulabelled size",len(unlabeled_indices))

    return labeled_indices,unlabeled_indices


class dataset(Dataset):

    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
 
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.length




def train_teacher_model(train_x,train_y):
    global teacher_model,model
    batch_size = 16
    no_of_batches = floor(train_x.shape[0]/batch_size)
    # opt = torch.optim.RMSprop(teacher_model.parameters(), lr=learning_rate)
    opt = torch.optim.SGD(teacher_model.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=0.0001)
    

    prog_bar = tqdm(range(no_of_batches))
    for batch_idx in prog_bar:
        
        for epoch in range(epochs):
            
            lab_X = torch.from_numpy(train_x[batch_idx*batch_size:batch_idx*batch_size+batch_size]).to(device)
            if image_resolution is not None:
                    lab_X = lab_X.reshape(image_resolution)
                   
            y_pred = model(lab_X)
            lab_y = torch.from_numpy(train_y[batch_idx*batch_size:batch_idx*batch_size+batch_size]).to(device).reshape(y_pred.shape)
            sup_loss = loss_fn(y_pred,lab_y.float())
            total_loss = sup_loss            
            opt.zero_grad()
            total_loss.backward()
            opt.step() 
    
    # model.load_state_dict(teacher_model.state_dict())

def compute_distill_loss(unlabeled_pre,unlabeled_x):
    global teacher_model,model

    if image_resolution is not None:
       unlabeled_x = unlabeled_x.reshape(image_resolution)

    unlabeled_gt = teacher_model(unlabeled_x)
    # unlabeled_gt = model(unlabeled_x)
    distillation_loss = loss_fn(unlabeled_gt,unlabeled_pre)

    return distillation_loss


def sample_batch_from_memory(mem_batchsize,minority_alloc):
    if mem_batchsize > 0:
        majority_class_idices,minority_class_indices = [],[]
        global memory_X,memory_y,memory_y_name,minorityclass_ids
        # minority_classes = [int(class_idx) for class_idx in minorityclass_ids]
        # unique_class = np.unique(memory_y_name).tolist()
        # majority_class = list(set(unique_class)-set(minority_classes))
    
        # for class_idx in majority_class:
        #     indices = (np.where(memory_y_name == int(class_idx))[0]).tolist()
        #     majority_class_idices.extend(indices)

    
        # minority_class_indices = list(set(range(0,memory_X.shape[0]))-set(majority_class_idices))
        # minority_offset = floor(mem_batchsize*minority_alloc)
        # majority_offset = mem_batchsize-minority_offset
        # select_indices = min(minority_offset,len(minority_class_indices))
        # select_indices = max(1, select_indices)
        # minority_class_indices = random.sample(minority_class_indices,select_indices)
        # select_indices = min(majority_offset,len(majority_class_idices))
        # select_indices = max(1, select_indices)
        # majority_class_idices = random.sample(majority_class_idices,select_indices)
        # minority_class_indices.extend(majority_class_idices)
        select_indices = min(mem_batchsize,memory_X.shape[0])
        minority_class_indices = random.sample(list(range(0,memory_X.shape[0])),select_indices)
    
    # print(minority_class_indices)
    

        return memory_X[minority_class_indices],memory_y[minority_class_indices],memory_y_name[minority_class_indices]
    
    


def train(tasks,task_class_ids,task_id,feature_list,threshold,X_val,y_val):
    global memory_X, memory_y, memory_y_name,local_count,global_count,local_store,input_shape,memory_size,task_num
    global classes_so_far,full,global_priority_list,local_priority_list,memory_population_time,replay_size
    global memory_population_time,epochs,grad_norm_dict,temp_norm

    grad_norm_list = []

    valid_loader = torch.utils.data.DataLoader(dataset(X_val,y_val),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=0)
    feature_mat = []
    X,y,y_classname = tasks[0][0],tasks[0][1],tasks[0][2]
    task_size = X.shape[0]
    labeled_indicies,unlabeled_indicies=split_a_task(tasks,labels_ratio,task_class_ids)
    

    labeled_X,labeled_y,labeled_y_classname = X[labeled_indicies],y[labeled_indicies],y_classname[labeled_indicies]
    X_unlab,y_unlab,y_unlabclassname = X[unlabeled_indicies],y[unlabeled_indicies],y_classname[unlabeled_indicies]
   
    
    
    if task_id > 0:
              
            mem_batch_size = floor(batch_size*b_m)
            rem_batch_size = batch_size-mem_batch_size
            # task_size = X.shape[0] + memory_X.shape[0] 
            labeled_batch_size = floor(rem_batch_size*labels_ratio)
            unlabeled_batch_size = rem_batch_size - (labeled_batch_size)
            no_of_batches = floor(len(labeled_indicies)/labeled_batch_size)
            no_of_unlab_batches = floor(len(unlabeled_indicies)/unlabeled_batch_size)
            p = np.random.permutation(labeled_X.shape[0])
            labeled_X,labeled_y,labeled_y_classname = labeled_X[p,:],labeled_y[p],labeled_y_classname[p]
            
    else:
        # initialize_buffermemory(labeled_task,memory_size)
        task_size = X.shape[0]    
        labeled_batch_size = floor(batch_size*labels_ratio)
        unlabeled_batch_size = batch_size-labeled_batch_size
        no_of_batches = floor(task_size/batch_size)

    if bool_gpm:
        for i in range(len(feature_list)):
            # print(feature_list[i].shape)
            Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
            feature_mat.append(Uf)    
    
    # prog_bar = tqdm(range(no_of_batches))
    # for batch_idx in prog_bar:
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    check_point_file_name = "checkpoint"+str(os.getpid())+".pt"
    check_point_file_name_norm = "checkpoint"+str(os.getpid())+"grad_norm"+".pt"
    early_stopping = EarlyStopping(patience=3, verbose=True,path=check_point_file_name)
    gradient_rejection = GradientRejection(patience=2, verbose=True,path=check_point_file_name_norm)
    scheduler = StepLR(opt, step_size=1, gamma=0.96)
    for epoch in range(epochs):
        # print("epoch",epoch)
        # scheduler.step()
        prog_bar = tqdm(range(no_of_batches))
        for batch_idx in prog_bar:
            model.train()        
        # for epoch in range(epochs):
            with torch.no_grad():
                if task_id > 0 and batch_idx < no_of_unlab_batches:
                    unlabeled_X = torch.from_numpy(X_unlab[batch_idx*unlabeled_batch_size:batch_idx*unlabeled_batch_size+unlabeled_batch_size]).to(device)
                else:
                    rand_indices = list(random.sample(range(X_unlab.shape[0]),min(unlabeled_batch_size,X_unlab.shape[0])))
                    unlabeled_X = torch.from_numpy(X_unlab[rand_indices]).to(device)


                if image_resolution is not None:
                    unlabeled_X = unlabeled_X.reshape(image_resolution)
                unlabeled_pred = model(unlabeled_X).detach()
            
            lab_X = labeled_X[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]  
            lab_y = labeled_y[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]
            if task_id > 0:
                
                mem_batch = sample_batch_from_memory(floor(batch_size*b_m),minority_alloc=batch_minority_alloc)
                if mem_batch is not None and mem_batch[0].shape[0] > 0:
                    
                    lab_X = np.concatenate((lab_X,mem_batch[0]), axis=0)  
                    temp_mem_X = torch.from_numpy(mem_batch[0]).to(device)
                    #if image_resolution is not None:
                     #   temp_mem_X = temp_mem_X.reshape(image_resolution)
                    #temp_mem_y = teacher_model(temp_mem_X).detach().cpu().numpy().squeeze()
                    
                    #lab_y = np.concatenate((lab_y,temp_mem_y), axis=0)
                    
                    lab_y = np.concatenate((lab_y,mem_batch[1]), axis=0)
            lab_X = torch.from_numpy(lab_X).to(device)
            if image_resolution is not None:
                    lab_X = lab_X.reshape(image_resolution)
                   
            y_pred = model(lab_X).squeeze()            
            lab_y = torch.from_numpy(lab_y).to(device)#.reshape(y_pred.shape)
            sup_loss = loss_fn(y_pred,lab_y.float())
            total_loss = sup_loss
            distil_loss = 0
            distil_loss = torch.as_tensor(distil_loss).to(device)
            opt.zero_grad()
            if task_id > 0:
                #computing the distillation loss
                distil_loss = compute_distill_loss(unlabeled_pred,unlabeled_X)
                total_loss = total_loss +  alpha *distil_loss
                
                #computing projection matrix and projecting the gradient
                if bool_gpm:
                    total_loss.backward()
                    # for i in range(len(feature_list)):
                    #     Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
                    #     feature_mat.append(Uf)
                    for k, (m,params) in enumerate(model.named_parameters()):
                        sz =  params.grad.data.size(0)
                        params.grad.data = torch.mul((params.grad.data - torch.mul(torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[k]).view(params.size()),1)), (1))  
            else:       
                total_loss.backward()

            
            # torch.nn.utils.clip_grad_norm(model.parameters(), min(1,temp_norm))
            # grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
            # norm = torch.cat(grads).norm().detach().cpu().numpy().item()
            # for k, (m,params) in enumerate(model.named_parameters()):
            #             sz =  params.grad.data.size(0)
            #             params.grad.data = torch.mul(torch.div(params.grad.data,norm),1)
            grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
            norm = torch.cat(grads).norm().detach().cpu().numpy().item()            
            grad_norm_dict.append(norm)
            # temp_norm = (1-labels_ratio)*norm
            
            opt.step() 
            gradient_rejection(model=model)
            if gradient_rejection.early_stop:
                torch.save(model.state_dict(), check_point_file_name_norm)
            train_losses.append(total_loss.item())

            y_pred = y_pred.detach().cpu().numpy()
            lab_y = lab_y.detach().cpu().numpy()
            
            # lr_precision, lr_recall, _ = precision_recall_curve(lab_y, y_pred,pos_label=1)
            # lr_auc_outlier =  auc(lr_recall, lr_precision)
            
        

            # lr_precision, lr_recall, _ = precision_recall_curve(lab_y, [1-x for x in y_pred],pos_label=0)
            # lr_auc_inliers =  auc(lr_recall, lr_precision)   
            # prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f} - PR-AUC(inliers): {:.2f} - PR_auc(outlier)_curve {:.3f}'.format(
            #      total_loss.item(), sup_loss.item(), distil_loss.item(), lr_auc_inliers,lr_auc_outlier ))
            # r_auc = roc_auc_score(lab_y, y_pred)
            prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f}'.format(
                 total_loss.item(), sup_loss.item(), distil_loss.item()))
        
        model.eval() # prep model for evaluation
        val_pred,val_gt = [],[]
        for data, target in valid_loader:
            pred = model(data.to(device)).reshape(target.shape)
            y_pred = pred.detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_gt.extend(target.detach().cpu().numpy().tolist())
        lr_precision, lr_recall, _ = precision_recall_curve(val_gt, val_pred,pos_label=1)
        lr_auc =  auc(lr_recall, lr_precision)
            # calculate the loss
            # loss = loss_fn(pred, target.to(device))
            # record validation loss
            # valid_losses.append(loss.item())
        # valid_losses.append(np.nan_to_num(lr_auc))
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        # valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        # avg_valid_losses.append(valid_loss)
        epoch_len = len(str(epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'PR-AUC (I): {lr_auc:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(lr_auc, model)
        if early_stopping.counter <1:
            scheduler.step()

        if early_stopping.early_stop:
            print("Early stopping")
            break
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(check_point_file_name))

    temp_x,temp_y,temp_yname = X[labeled_indicies,:],y[labeled_indicies],y_classname[labeled_indicies]
    # temp_x,temp_y,temp_yname = X[unlabeled_indicies,:],y[unlabeled_indicies],y_classname[unlabeled_indicies]
    
    if task_id > 0:
        mem_start_time = time.time()
        if str(mem_strat) == "replace":
            
            tasks[0] = temp_x,temp_y,temp_yname
            lab_samples_in_memory = split_a_task(tasks,lab_samp_in_mem_ratio)
            tasks[0] = temp_x[lab_samples_in_memory[0],:],temp_y[lab_samples_in_memory[0]],temp_yname[lab_samples_in_memory[0]]
            initialize_buffermemory(tasks=tasks,mem_size=memory_size)
        elif str(mem_strat) == "equal":
            
            memory_X, memory_y, memory_y_name = memory_update_equal_allocation2(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.15,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)
        else:
            
            memory_X, memory_y, memory_y_name = memory_update_equal_allocation(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.15,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)

        # 
        # 
        # 
        # 
        mem_finish_time = time.time()
        memory_population_time += mem_finish_time-mem_start_time

    # mat_list = []    
    temp_x,temp_y,temp_yname = X[labeled_indicies,:],y[labeled_indicies],y_classname[labeled_indicies]
    # mat_list = get_representation_matrix (model, device, temp_x, temp_y)
    if bool_gpm:
        mat_list = get_representation_matrix (model, device, temp_x, temp_y,rand_samples=no_of_rand_samples)
        feature_list = update_GPM(model, mat_list, threshold, feature_list) 
      
    else:
        feature_list = []

    # grad_norm_dict[task_id] = grad_norm_list   
    # print(grad_norm_dict)
    if os.path.exists(check_point_file_name):
        os.remove(check_point_file_name)
    if os.path.exists(check_point_file_name_norm):
        os.remove(check_point_file_name_norm) 

    
    return feature_list

     





# def train(tasks,task_class_ids,task_id,feature_list,threshold):


#     feature_mat = []
    
    
        

#     # print(type(tasks))
#     alpha=9


#     X,y,y_classname = tasks[0][0],tasks[0][1],tasks[0][2]

#     global memory_X, memory_y, memory_y_name,local_count,global_count,local_store,input_shape,memory_size,task_num
#     global classes_so_far,full,global_priority_list,local_priority_list,memory_population_time,replay_size
#     global ecbrs_taskaware_memory_X ,ecbrs_taskaware_memory_y,ecbrs_taskaware_memory_y_name,memory_population_time
#     # task_id_temp = 0
#     # labels_ratio=0.1
#     # no_of_rand_samples = 1000
#     # minority_alloc = 0.8
#     labeled_indicies,unlabeled_indicies=split_a_task(tasks,labels_ratio,task_class_ids)
    

#     labeled_X,labeled_y,labeled_y_classname = X[labeled_indicies],y[labeled_indicies],y_classname[labeled_indicies]
#     X_unlab,y_unlab,y_unlabclassname = X[unlabeled_indicies],y[unlabeled_indicies],y_classname[unlabeled_indicies]
   
    
#     task_size = X.shape[0]
#     if task_id > 0:
#             mem_batch_size = floor(batch_size*b_m)
#             print("mem batch size",mem_batch_size)
#             rem_batch_size = batch_size-mem_batch_size
#             # task_size = X.shape[0] + memory_X.shape[0] 
#             labeled_batch_size = floor(rem_batch_size*labels_ratio)
#             unlabeled_batch_size = rem_batch_size - (labeled_batch_size)
#             no_of_batches = floor(len(labeled_indicies)/labeled_batch_size)
#             no_of_unlab_batches = floor(len(unlabeled_indicies)/unlabeled_batch_size)
#             p = np.random.permutation(labeled_X.shape[0])
#             labeled_X,labeled_y,labeled_y_classname = labeled_X[p,:],labeled_y[p],labeled_y_classname[p]
            
#     else:
#         # initialize_buffermemory(labeled_task,memory_size)
#         task_size = X.shape[0]    
#         labeled_batch_size = floor(batch_size*labels_ratio)
#         unlabeled_batch_size = batch_size-labeled_batch_size
#         no_of_batches = floor(task_size/batch_size)

#     # labeled_batch_size = floor(batch_size*labels_ratio)
    


   
  
#     prog_bar = tqdm(range(no_of_batches))
#     for batch_idx in prog_bar:
        
#         for epoch in range(epochs):
            
            
#             with torch.no_grad():
#                 if task_id > 0 and batch_idx < no_of_unlab_batches:
#                     unlabeled_X = torch.from_numpy(X_unlab[batch_idx*unlabeled_batch_size:batch_idx*unlabeled_batch_size+unlabeled_batch_size]).to(device)
#                 else:
#                     rand_indices = list(random.sample(range(X_unlab.shape[0]),min(unlabeled_batch_size,X_unlab.shape[0])))
#                     unlabeled_X = torch.from_numpy(X_unlab[rand_indices]).to(device)
                
#                 if image_resolution is not None:
#                     unlabeled_X = unlabeled_X.reshape(image_resolution)
#                 unlabeled_pred = model(unlabeled_X).detach()
            
#             lab_X = labeled_X[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]  
#             lab_y = labeled_y[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]
#             if task_id > 0:
#                 mem_batch = sample_batch_from_memory(floor(batch_size*b_m),batch_minority_alloc)
#                 if mem_batch is not None and mem_batch.shape[0]>0:
#                     lab_X = np.concatenate((lab_X,mem_batch[0]), axis=0)  
#                     lab_y = np.concatenate((lab_y,mem_batch[1]), axis=0)
#             lab_X = torch.from_numpy(lab_X).to(device)
            
#             if image_resolution is not None:
#                     lab_X = lab_X.reshape(image_resolution)
                   
#             y_pred = model(lab_X)            
#             lab_y = torch.from_numpy(lab_y).to(device).reshape(y_pred.shape)
#             sup_loss = loss_fn(y_pred,lab_y.float())
#             total_loss = sup_loss
#             distil_loss = 0
#             distil_loss = torch.as_tensor(distil_loss).to(device)
#             opt.zero_grad()
#             if task_id > 0:
#                 #computing the distillation loss
#                 distil_loss = compute_distill_loss(unlabeled_pred,unlabeled_X)
#                 total_loss = total_loss + alpha * distil_loss

#                 #computing projection matrix and projecting the gradient
               
#                 for i in range(len(model.act)):
#                     Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
#                     feature_mat.append(Uf)
#                 # print ('-'*40)
#                 total_loss.backward()
#                 for k, (m,params) in enumerate(model.named_parameters()):
#                     sz =  params.grad.data.size(0)
#                     params.grad.data = params.grad.data - torch.mul(torch.mm(params.grad.data.view(sz,-1),\
#                                                     feature_mat[k]).view(params.size()),1)
#             # opt.zero_grad()
#             else:
#                 total_loss.backward()
            
#             opt.step() 

#             y_pred = y_pred.detach().cpu().numpy()
#             lab_y = lab_y.detach().cpu().numpy()
            
#             lr_precision, lr_recall, _ = precision_recall_curve(lab_y, y_pred,pos_label=1)
#         # calculate scores
#             lr_auc_outlier =  auc(lr_recall, lr_precision)
#             # summarize scores
        

#             lr_precision, lr_recall, _ = precision_recall_curve(lab_y, [1-x for x in y_pred],pos_label=0)
#         # calculate scores
#             lr_auc_inliers =  auc(lr_recall, lr_precision)   
#             prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f} - PR-AUC(inliers): {:.2f} - PR_auc(outlier)_curve {:.3f}'.format(
#                  total_loss.item(), sup_loss.item(), distil_loss.item(), lr_auc_inliers,lr_auc_outlier ))
        

#     temp_x,temp_y,temp_yname = X[labeled_indicies,:],y[labeled_indicies],y_classname[labeled_indicies]
        
 
#     if task_id >= 0:
#         mem_start_time = time.time()
#         # memory_X, memory_y, memory_y_name = memory_update_equal_allocation(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.15,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)
#         tasks[0] = temp_x,temp_y,temp_yname
#         initialize_buffermemory(tasks=tasks,mem_size=memory_size)
#         mem_finish_time = time.time()
#         memory_population_time += mem_finish_time-mem_start_time

#     mat_list = get_representation_matrix (model, device, temp_x, temp_y,rand_samples=no_of_rand_samples)
#     feature_list = update_GPM(model, mat_list, threshold, feature_list)    
#     #feature_list = []
    
#     return feature_list
     
def get_whole_test_set():
    global test_x,test_y,task_num,teacher_model,model,task_order,X_test,y_test
    test_x,test_y = [],[]
    task_order1 = task_order
    # random.shuffle(task_order1)
    for task_id,task in enumerate(task_order1):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        # print("loading task:",task_id)     
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=bool_encode_benign,bool_encode_anomaly=bool_encode_anomaly,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=load_whole_train_data)
        test_x.extend([X_test])
        test_y.extend([y_test])

    X_test,y_test = np.concatenate( test_x, axis=0 ),np.concatenate( test_y, axis=0 )

def taskwise_lazytrain():
    global test_x,test_y,task_num,teacher_model,model,task_order,auc_result
    # random.shuffle(task_order)
    print("task order",task_order)
    threshold = np.array([0.95,0.99,0.99,0.98,0.99,0.99,0.99])
    feature_list =[]
    for task_id,task in enumerate(task_order):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("loading task:",task_id)     
        input_shape,tasks,X_test,y_test,X_val,y_val = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=bool_encode_benign,bool_encode_anomaly=bool_encode_anomaly,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=load_whole_train_data)
        test_x.extend([X_test])
        test_y.extend([y_test])
        print("Training task:",task_id)
        task_num = task_id
        if task_num == int(0):
            initialize_buffermemory(tasks=tasks,mem_size=memory_size)
        if task_id < 3:
            feature_list =train(tasks,task_class_ids,task_id,feature_list,threshold,X_val,y_val)
        # for i in range(len(mat_list)):
        #     print("each layer_size",mat_list[i].shape)
        # if bool_gpm:
        #     feature_list = update_GPM(model, mat_list, threshold, feature_list) 
        # # initialize_buffermemory(tasks=tasks,mem_size=memory_size)
        teacher_model.load_state_dict(model.state_dict())
        # evaluate_on_testset()
        
    #     auc_result[str(args.seed)+"_"+str(task_num)] = evaluate_on_testset()
    # with open(temp_filename, 'w') as fp:
    #     json.dump(auc_result, fp)
        

def evaluate_on_sub_testset(test_x,test_y):
    test_x,test_y = np.concatenate( test_x, axis=0 ),np.concatenate( test_y, axis=0 )
    model.eval()
    print("computing the results")
    offset = 25000
    for idx in range(0,test_x.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(test_x[idx1:idx2,:].astype(float)).to(device)
        if image_resolution is not None:
                    X_test1 = X_test1.reshape(image_resolution)
        temp = model(X_test1.float()).detach().cpu().numpy()
        if idx1==0:
            yhat = temp
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
    compute_results(test_y,yhat)
    model.train()

def evaluate_on_testset():
    
    global X_test,y_test
    if pth_testset is not None:
        X_test,y_test = load_teset(pth_testset,testset_class_ids,label)
    yhat = None    
    model.eval()
    print("computing the results")
    offset = 250000
    # offset = 25000
    for idx in range(0,X_test.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(X_test[idx1:idx2,:].astype(float)).to(device)
        if image_resolution is not None:
                    X_test1 = X_test1.reshape(image_resolution)
        temp = model(X_test1.float()).detach().cpu().numpy()
        if idx1==0:
            yhat = temp
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
    return compute_results(y_test,yhat)
    # print("test sample counters are",Counter(y_test))


def tsne_visualize(seed,labels_ratio=0.1,batch_minority=0.5,rand_samples=100,ppt=50):
    global X_test,y_test
    test_embeddings = torch.zeros((0,10), dtype=torch.float32)
    if pth_testset is not None:
        X_test,y_test = load_teset(pth_testset,testset_class_ids,label)
    yhat = None    
    model.eval()
    print("computing the results")
    offset = 25000
    for idx in range(0,X_test.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(X_test[idx1:idx2,:].astype(float)).to(device)
        if image_resolution is not None:
                    X_test1 = X_test1.reshape(image_resolution)
        temp = model(X_test1.float()).detach().cpu().numpy()
        embeddings = model.act['last_layer_activation']
        if idx1==0:
            yhat = temp
            
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
        test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)    
    test_embeddings = np.array(test_embeddings)
    dir_struct = {0:"tsne",1:"caring",2:str(label)}    
    dir_struct[3 ]= "_lab_ratio_"+str(labels_ratio)+"_minorty_"+str(batch_minority)+"_rand_samp_"+str(rand_samples)+"_seed"+str(seed)
    plot_tsne(y_test,yhat,test_embeddings,dir_struct,ppt)

def plot_grdient_norm_line_graph():
    dir_struct = {0:"line_graph",1:"caring",2:str(label)}    
    dir_struct[3 ]= "_lab_ratio_"+str(labels_ratio)+"_minorty_"+str(b_m)+"_bool_gpm"+str(bool_gpm)+"_rand_samp_"+str(no_of_rand_samples)+"_seed"+str(seed)
    plot_grad_norm_line_graph(dir_struct,grad_norm_dict)




def start_execution(dataset_name,l_rate,w_decay):
    global input_shape,tasks,X_test,y_test,test_x,test_y
    start_time=time.time()
    load_metadata(dataset_name,l_rate,w_decay)
    # load_model_metadata()
    # print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters is",pytorch_total_params)
    if is_lazy_training:
        test_x,test_y = [],[]
        # get_whole_test_set()
        taskwise_lazytrain()
        # plot_grdient_norm_line_graph()
        X_test,y_test = np.concatenate( test_x, axis=0 ),np.concatenate( test_y, axis=0 )
        

    else:
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,class_ids,minorityclass_ids,tasks_list,task2_list,task_order,bool_encode_benign=False,bool_encode_anomaly=True,label=label,bool_create_tasks_avalanche=False)
        initialize_buffermemory(tasks=tasks,mem_size=memory_size)
        print('Total no.of tasks', len(tasks))
        # update_buffermemory_counter(memorysamples=memory_y_name)
        # update_mem_samples_indexdict(memorysamples=memory_y_name)
        train(tasks=tasks)
    print("Total execution time is--- %s seconds ---" % (time.time() - start_time))
    print("Total memory population time is--- %s seconds ---" % (memory_population_time))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--ds', type=str, default="cifar100", metavar='S',help='dataset name')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)')
    parser.add_argument('--filename', type=str,default="temp", metavar='S',help='json file name')
    parser.add_argument('--b_m', type=float, default=0.2, metavar='S',help='batch memory ratio(default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='S',help='learning rate(default: 0.001)')
    parser.add_argument('--wd', type=float, default=0.001, metavar='S',help='weight decay(default: 0.01)')
    parser.add_argument('--label_ratio', type=float, default=0.05, metavar='S',help='labeled ratio (default: 0.1)')
    parser.add_argument('--nps', type=int, metavar='S',default=100,help='number of projection samples(default: 100)')
    parser.add_argument('--bma', type=float, metavar='S',default=0.8,help='batch minority allocation(default: 0.8)')
    parser.add_argument('--alpha', type=float, metavar='S',default=9,help='distill loss multiplier(default: 9)')
    parser.add_argument('--lab_samp_in_mem_ratio', type=float, metavar='S',default=0.1,help='Percentage of labeled samples to store in memory(default: 1.0)')
    parser.add_argument('--bool_gpm', type=str, metavar='S',default="True",help='Enables gradient projections(default: True)')
    parser.add_argument('--mem_strat', type=str, metavar='S',default="replace",help='Buffer memory strategy(default: full initialization)')


    args = parser.parse_args()
    set_seed(args.seed)
    get_gpu(args.gpu)
    print("seed is",args.seed)
    global labels_ratio,no_of_rand_samples,l_rate,w_decay,batch_minority_allocation,b_m,alpha,lab_samp_in_mem_ratio,bool_gpm,mem_strat,temp_filename,auc_result,seed
    b_m = float(args.b_m)
    labels_ratio=float(args.label_ratio)
    no_of_rand_samples = int(args.nps)
    batch_minority_alloc = float(args.bma)
    alpha = float(args.alpha)
    l_rate = float(args.lr)
    w_decay = float(args.wd)
    lab_samp_in_mem_ratio = float(args.lab_samp_in_mem_ratio)
    bool_gpm = eval(args.bool_gpm)
    mem_strat = str(args.mem_strat)
    seed = args.seed
    ppt = 50
    print("{:<20}  {:<20}".format('Argument','Value'))
    print("*"*80)
    for arg in vars(args):
        print("{:<20}  {:<20}".format(arg, getattr(args, arg)))
    print("*"*80)    
    auc_result= {}
    temp_filename = str(args.filename)    
    start_execution(args.ds,l_rate,w_decay)
    print("seed is",args.seed)
    grad_norm_mean = sum(grad_norm_dict)/len(grad_norm_dict)
    grad_norm_variance = statistics.variance(grad_norm_dict)
    print("grad norm avg:",grad_norm_mean)
    print("grad norm variance",grad_norm_variance)
    
    with open(temp_filename, 'w') as fp:
        test_set_results = evaluate_on_testset()
        test_set_results.extend([grad_norm_mean,grad_norm_variance])
        auc_result[str(args.seed)] = test_set_results
        json.dump(auc_result, fp)
    
    print("*"*80)
    
    print("grad norm avg:",grad_norm_mean)
    print("grad norm variance",grad_norm_variance)
    
    # tsne_visualize(args.seed,labels_ratio,batch_minority_alloc,no_of_rand_samples,ppt)


    

