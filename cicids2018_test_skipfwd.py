

from turtle import st
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torch.optim.lr_scheduler import StepLR
from scipy.spatial.distance import cdist
from scipy import stats
# from pytorchtools import EarlyStopping
import subprocess
import os
import tempfile
import numpy as np
import pandas as pd
import statistics
import math
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

from utils.otdd.ot_distance import compute_ot_distance



import time
import random
from math import floor
from collections import Counter
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,roc_curve
from sklearn.metrics import f1_score,confusion_matrix
from tqdm import tqdm
import itertools
import argparse
import json
import multiprocessing as mp
# mp.set_start_method('spawn')
from tabulate import tabulate


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
replay_size,memory_size,minority_allocation,batch_size,device,pattern_per_exp,is_lazy_training,task_num = None,None,None,None,None,None,None,None
memory_X, memory_y, memory_y_name,ecbrs_taskaware_memory_X, ecbrs_taskaware_memory_y,ecbrs_taskaware_memory_y_name,memory_per_task = None,None,None,None,None,None,None
loss_fn,train_acc_metric = None,None
student_model1,student_model2,student_supervised,student_optimizer1,student_optimizer2,student_supervised_optimizer = None,None,None,None,None,None
teacher_model1,teacher_model2,teacher_supervised =None,None,None
pth_testset,testset_class_ids =None,None
test_x,test_y = None,None
val_x_all_tasks,val_y_all_tasks = None, None
image_resolution = None
bool_encode_anomaly,bool_encode_benign,load_whole_train_data=None,None,None
nc = 0
no_tasks = 0
grad_norm_dict = []
temp_norm =1
#train_with_unlab=True
# labels_ratio, no_of_rand_samples, minority_alloc = None,None,None

# consecutive_otdd = []
owl_self_labelled_count_class_0 = 0
owl_self_labelled_count_class_1 = 0
owl_analyst_labelled_count_class_0 = 0
owl_analyst_labelled_count_class_1 = 0
# BUGFIX (ported from cicids2017_spider_owl_neurips2024_2.py -- see that file's
# history for the full trace): must start as None, not 0, so the `is None`
# calibration-update branches below can ever fire.
truth_agreement_fraction_0, truth_agreement_fraction_1 = None, None

# SELF-LABEL ACCURACY tracking: one entry per unseen (OWL self-labeled)
# task, appended inside owl_data_labeling_strategy() when unseen_task=True.
# Ground truth is used here for measurement/reporting only, never fed back
# into training.
SELF_LABEL_ACCURACY_LOG = []

CI_list = []
avg_CI = None
adp_attack_cos_dist,adp_benign_cos_dist = 0,0



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
    batch_size = ds.batch_size
   
    device = cfg.device
    learning_rate = l_rate# 0.001#ds.learning_rate
    no_tasks = ds.no_tasks
    pattern_per_exp = ds.pattern_per_exp
    is_lazy_training = ds.is_lazy_training
    ecbrs_taskaware = ds.taskaware_ecbrs
    bool_encode_anomaly = False
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
    global student_model1,student_model2,student_supervised,student_optimizer1,student_optimizer2,student_supervised_optimizer,loss_fn,train_acc_metric,input_shape
    global teacher_model1,teacher_model2,teacher_supervised

    w_d = w_decay
    student_model1 = load_model(label=label,inputsize=get_inputshape(pth,class_ids))
    print(student_model1)
    teacher_model1 = load_model(label=label,inputsize=get_inputshape(pth,class_ids))
    student_optimizer1 = torch.optim.SGD(student_model1.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=w_d)
    student_model1 = student_model1.to(device)
    teacher_model1 = teacher_model1.to(device)

    if mlps >=2:
        student_model2 = load_model(label=label+"_student2",inputsize=get_inputshape(pth,class_ids))
        teacher_model2 = load_model(label=label+"_student2",inputsize=get_inputshape(pth,class_ids))
        student_optimizer2 = torch.optim.SGD(student_model2.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=w_d)
        student_model2 = student_model2.to(device)
        teacher_model2 = teacher_model2.to(device)
    if mlps == 3:
        student_supervised = load_model(label=label+"_supervised",inputsize=get_inputshape(pth,class_ids))      
        teacher_supervised = load_model(label=label+"_supervised",inputsize=get_inputshape(pth,class_ids))
        student_supervised_optimizer = torch.optim.SGD(student_supervised.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=w_d)
        student_supervised = student_supervised.to(device)
        teacher_supervised = teacher_supervised.to(device)

    
    
    
    
    loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.CrossEntropyLoss()
    # train_acc_metric = Accuracy(task='multiclass',                                           
    #                                  num_classes=2).to(device)

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
    # benign_indices = benign_indices[0:b_min_length]
    # benign_indices.extend(attack_indices[0:a_min_length])
    benign_indices = attack_indices[0:a_min_length]
    
                                    

    
    b=benign_indices
    rand_samples = len(benign_indices)
    x=torch.tensor(x,dtype=torch.float32).to(device)
    example_data = x[b]#.view(-1,70)
    example_data = example_data.to(device)
    example_out  = net(example_data)
    
    # batch_list=[int(rand_samples),int(rand_samples),int(rand_samples),int(rand_samples),int(rand_samples),rand_samples,rand_samples] 
    mat_list=[] # list contains representation matrix of each layer
    act_key=list(net.act.keys())
    batch_list = [int(rand_samples) for idx_keys in range(len(act_key))]
    print("keys are",act_key)

    for i in range(len(act_key)):
        # print(act_key[i])
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


def compute_distill_loss(unlabeled_pre,unlabeled_x):
    
    global teacher_model1,teacher_model2,teacher_supervised,student_model1,student_model2,student_supervised

    if image_resolution is not None:
       unlabeled_x = unlabeled_x.reshape(image_resolution)

    if mlps == 1:
        models = [student_model1]
    elif mlps == 2:
        models = [student_model1,student_model2]
    else:
        models = [student_model1,student_model2,student_supervised]    

    #models = [teacher_model1,teacher_model2]#,teacher_supervised] 
    
    class_probs = []  

    with torch.no_grad():
        for model in models:
            outputs = torch.softmax(model(unlabeled_x), dim=1)
            class_probs.append(outputs)

    avg_probs = torch.stack(class_probs).mean(dim=0)
    predicted_labels = torch.argmax(avg_probs, dim=1) 
    unlabeled_gt = F.one_hot(predicted_labels,2)
    # unlabeled_gt = model(unlabeled_x)
    distillation_loss = loss_fn(unlabeled_pre.float(),unlabeled_gt.float())

    return [distillation_loss,predicted_labels]

# def compute_distill_loss_with_confidence(unlabeled_pre,unlabeled_x):
    
#     global teacher_model1,teacher_model2,teacher_supervised,student_model1,student_model2,student_supervised

#     if image_resolution is not None:
#        unlabeled_x = unlabeled_x.reshape(image_resolution)

#     if mlps == 1:
#         models = [student_model1]
#     elif mlps == 2:
#         models = [student_model1,student_model2]
#     else:
#         models = [student_model1,student_model2,student_supervised]    

#     #models = [teacher_model1,teacher_model2]#,teacher_supervised] 
    
#     class_probs = []  

#     with torch.no_grad():
#         for model in models:
#             outputs = torch.softmax(model(unlabeled_x), dim=1)
#             class_probs.append(outputs)

#     avg_probs = torch.stack(class_probs).mean(dim=0)
#     max_probs, _ = torch.max(avg_probs, dim=1)

# # 2. Confidence threshold
#     mask = max_probs > 0.95

# # 3. Safeguard: proceed only if confident samples exist
#     if mask.any():
#         # 4. Restrict to confident samples ONLY
#         confident_probs = avg_probs[mask]
#         confident_unlabeled_pre = unlabeled_pre[mask]

#         # 5. Argmax ONLY on confident samples
#         predicted_labels = torch.argmax(confident_probs, dim=1)

#         # 6. One-hot pseudo-labels
#         unlabeled_gt = F.one_hot(predicted_labels, num_classes=2)

#         # 7. Distillation loss
#         distillation_loss = loss_fn(
#             confident_unlabeled_pre.float(),
#             unlabeled_gt.float()
#     )

#     else:

#     # No confident samples → no gradient contribution
#         distillation_loss = torch.tensor(
#         0.0, device=avg_probs.device, requires_grad=False
#         )
    

#     return [distillation_loss,predicted_labels]



def compute_distill_loss_with_confidence(unlabeled_pre,unlabeled_x):
    
    global teacher_model1,teacher_model2,teacher_supervised,student_model1,student_model2,student_supervised

    predicted_labels,confident_unlabeled_pre = None,None

    if image_resolution is not None:
       unlabeled_x = unlabeled_x.reshape(image_resolution)

    if mlps == 1:
        models = [student_model1]
    elif mlps == 2:
        models = [student_model1,student_model2]
    else:
        models = [student_model1,student_model2,student_supervised]    

    #models = [teacher_model1,teacher_model2]#,teacher_supervised] 
    
    class_probs = []  

    with torch.no_grad():
        for model in models:
            outputs = torch.softmax(model(unlabeled_x), dim=1)
            class_probs.append(outputs)

    avg_probs = torch.stack(class_probs).mean(dim=0)
    max_probs, _ = torch.max(avg_probs, dim=1)

# 2. Confidence threshold
    mask = max_probs > 0.95

# 3. Safeguard: proceed only if confident samples exist
    if mask.any():
        # 4. Restrict to confident samples ONLY
        confident_probs = avg_probs[mask]
        confident_unlabeled_pre = unlabeled_pre[mask]

        # 5. Argmax ONLY on confident samples
        predicted_labels = torch.argmax(confident_probs, dim=1)

        # 6. One-hot pseudo-labels
        unlabeled_gt = F.one_hot(predicted_labels, num_classes=2)

        # 7. Distillation loss
        distillation_loss = loss_fn(
            confident_unlabeled_pre.float(),
            unlabeled_gt.float()
    )

    else:

    # No confident samples → no gradient contribution
        distillation_loss = torch.tensor(
        0.0, device=avg_probs.device, requires_grad=False
        )
    

    return [distillation_loss,predicted_labels,confident_unlabeled_pre]




# def compute_distill_loss(unlabeled_pre,unlabeled_x):
    
#     global teacher_model1,teacher_model2,teacher_supervised,student_model1,student_model2,student_supervised,memory_X

#     if image_resolution is not None:
#        unlabeled_x = unlabeled_x.reshape(image_resolution)

#     if mlps == 1:
#         models = [student_model1]
#     elif mlps == 2:
#         models = [student_model1,student_model2]
#     else:
#         models = [student_model1,student_model2,student_supervised]    

#     #models = [teacher_model1,teacher_model2]#,teacher_supervised] 
    
#     class_probs = []  

#     with torch.no_grad():
#         for model in models:
#             outputs = torch.softmax(model(unlabeled_x), dim=1)
#             class_probs.append(outputs)

#     avg_probs = torch.stack(class_probs).mean(dim=0)
#     predicted_labels = torch.argmax(avg_probs, dim=1) 
#     unlabeled_gt = F.one_hot(predicted_labels,2)
#     # unlabeled_gt = model(unlabeled_x)

#     mem_X_norm = np.linalg.norm(memory_X, axis=1, keepdims=True)
#     arr1 = memory_X / mem_X_norm
#     unlabeled_x = unlabeled_x.detach().cpu().numpy()
#     unlabeled_pre = unlabeled_pre.detach().cpu().numpy()
#     unlab_X_norm = np.linalg.norm(unlabeled_x, axis=1, keepdims=True)
#     arr2 = unlabeled_x / unlab_X_norm
#     cos_sim = np.dot(arr2,arr1.T)
#     label_frequency = []
#     label1_indices = (np.where(memory_y == 1)[0]).tolist()    
#     label0_indices = (np.where(memory_y == 0)[0]).tolist()
#     label_frequency.append(len(label0_indices))
#     label_frequency.append(len(label1_indices))
#     weight_matrix = np.zeros_like(cos_sim)
#     for col_idx in range(weight_matrix.shape[1]):
#         weight_matrix[:,col_idx] = 1/label_frequency[memory_y[col_idx]]

        
#     cos_sim = cos_sim * weight_matrix
#     # mem_decision_softmax = np.zeros_like(unlabeled_pre)
#     for row_idx in range(cos_sim.shape[0]):
#         label0_prob = np.sum(np.exp(cos_sim[row_idx,label0_indices]))/np.sum(np.exp(cos_sim[row_idx,:]))
#         # print("hello",label0_prob)
#         # print("hello3",unlabeled_pre[row_idx])
#         # label1_prob = 1-label0_prob#np.sum(np.exp(cos_sim[row_idx,label1_indices]))/np.sum(np.exp(cos_sim[i,:]))
#         # mem_decision_softmax[row_idx,0],mem_decision_softmax[row_idx,1] = label0_prob,1-label0_prob
#         max_index_1 = np.argmax(unlabeled_pre[row_idx])
#         # max_index_2 = np.argmax(mem_decision_softmax[row_idx])
#         # max_index_2 = np.argmax([label0_prob,1-label0_prob]])
#         if unlabeled_pre[row_idx,max_index_1] < max(label0_prob,1-label0_prob):#mem_decision_softmax[row_idx,max_index_2]:
#             # unlabeled_pre[row_idx] = mem_decision_softmax[row_idx]
#             unlabeled_pre[row_idx,0],unlabeled_pre[row_idx,1] = label0_prob,1-label0_prob
#             # print("hello2",unlabeled_pre[row_idx])
#             # exit()

#     unlabeled_x = torch.from_numpy(unlabeled_x).to(device)
#     unlabeled_pre = torch.from_numpy(unlabeled_pre).to(device)
#     distillation_loss = loss_fn(unlabeled_pre.float(),unlabeled_gt.float())

#     return [distillation_loss,predicted_labels]


# Contrastive loss function

def contrastive_loss(anchor_representations, positive_representations, negative_representations, temperature=0.5):
    # Combine representations for efficient matrix operations
    # print(anchor_representations.shape,positive_representations.shape,negative_representations.shape)
    all_representations = torch.cat([anchor_representations, positive_representations, negative_representations], dim=0)

    # Calculate cosine similarities in a single matrix operation
    similarities = torch.mm(all_representations, all_representations.T) / temperature

    # Extract relevant similarities efficiently
    positive_similarities = similarities[:anchor_representations.shape[0], anchor_representations.shape[0]:anchor_representations.shape[0] * 2]
    negative_similarities = similarities[:anchor_representations.shape[0], anchor_representations.shape[0] * 2:]

    # Calculate loss with optimized operations
    log_positive_similarities = torch.log(positive_similarities + 1e-8)
    negative_loss = torch.logsumexp(negative_similarities, dim=-1).mean()
    loss = -log_positive_similarities.mean() + negative_loss

    return loss

#

def construct_positive_negative_samples(batch_data, batch_labels):
    batch_size = batch_data.shape[0]

    # Pre-compute indices for positive and negative samples for each data point
    pos_neg_indices = torch.zeros((batch_size, 2), dtype=torch.long)
    for i in range(batch_size):
        
        same_class_indices = (batch_labels == batch_labels[i]).nonzero(as_tuple=True)[0]
        different_class_indices = (batch_labels != batch_labels[i]).nonzero(as_tuple=True)[0]
        pos_neg_indices[i, 0] = torch.randint(len(same_class_indices), (1,))[0]
        pos_neg_indices[i, 1] = torch.randint(len(different_class_indices), (1,))[0]

    # Get positive and negative samples directly using pre-computed indices
    positive_samples = batch_data[pos_neg_indices[:, 0]]
    negative_samples = batch_data[pos_neg_indices[:, 1]]

    # print("sample shape",positive_samples.shape,negative_samples.shape)
    return positive_samples, negative_samples

def construct_positive_negative_samples_from_memory(batch_labels):
    batch_size = batch_labels.shape[0]
    memory_X_tensor,memory_y_tensor = torch.from_numpy(memory_X).to(device),torch.from_numpy(memory_y).to(device)

    # Pre-compute indices for positive and negative samples for each data point
    pos_neg_indices = torch.zeros((batch_size, 2), dtype=torch.long)
    # print("mem_",memory_y)
    # print(batch_labels)
    for i in range(batch_size):
        
        same_class_indices = (memory_y_tensor == batch_labels[i]).nonzero(as_tuple=True)[0]
        different_class_indices = (memory_y_tensor != batch_labels[i]).nonzero(as_tuple=True)[0]
        pos_neg_indices[i, 0] = torch.randint(len(same_class_indices), (1,))[0]
        pos_neg_indices[i, 1] = torch.randint(len(different_class_indices), (1,))[0]

    # Get positive and negative samples directly using pre-computed indices
    positive_samples = memory_X_tensor[pos_neg_indices[:, 0]]
    negative_samples = memory_X_tensor[pos_neg_indices[:, 1]]

    # print("sample shape",positive_samples.shape,negative_samples.shape)
    return positive_samples, negative_samples    



def sample_batch_from_memory(mem_batchsize,minority_alloc):
    if mem_batchsize > 0:
        majority_class_idices,minority_class_indices = [],[]
        global memory_X,memory_y,memory_y_name,minorityclass_ids
        minority_classes = [int(class_idx) for class_idx in minorityclass_ids]
        unique_class = np.unique(memory_y_name).tolist()
        majority_class = list(set(unique_class)-set(minority_classes))
    
        for class_idx in majority_class:
            indices = (np.where(memory_y_name == int(class_idx))[0]).tolist()
            majority_class_idices.extend(indices)

    
        minority_class_indices = list(set(range(0,memory_X.shape[0]))-set(majority_class_idices))
        minority_offset = floor(mem_batchsize*minority_alloc)
        majority_offset = mem_batchsize-minority_offset
        select_indices = min(minority_offset,len(minority_class_indices))
        select_indices = max(1, select_indices)
        minority_class_indices = random.sample(minority_class_indices,select_indices)
        select_indices = min(majority_offset,len(majority_class_idices))
        select_indices = max(1, select_indices)
        # print(majority_class_idices,select_indices)
        majority_class_idices = random.sample(majority_class_idices,select_indices)
        minority_class_indices.extend(majority_class_idices)
        # select_indices = min(mem_batchsize,memory_X.shape[0])
        # minority_class_indices = random.sample(list(range(0,memory_X.shape[0])),select_indices)
    
    # print(minority_class_indices)
    

        return memory_X[minority_class_indices],memory_y[minority_class_indices],memory_y_name[minority_class_indices]
    
    
# def owl_data_labeling_strategy(X, y, y_classname, unseen_task=True):

#     global owl_self_labelled_count_class_0, owl_self_labelled_count_class_1
#     global owl_analyst_labelled_count_class_0, owl_analyst_labelled_count_class_1
#     global truth_agreement_fraction_0,truth_agreement_fraction_1
#     global avg_CI

#     print(f'X shape: {X.shape}')
#     dummy_target_label = torch.zeros(X.shape[0])
#     data_loader = torch.utils.data.DataLoader(dataset(X,dummy_target_label),
#                                             batch_size=batch_size,
#                                             #    sampler=valid_sampler,
#                                             num_workers=0)
        
#     predictions,predicted_labels = None,None

#     if mlps == 1:
#         models = [student_model1]
#     elif mlps == 2:
#         models = [student_model1,student_model2]
#     else:
#         models = [student_model1,student_model2,student_supervised]
    
#     for data, _ in data_loader:
#         class_probs = [] 
#         with torch.no_grad():
#             for model in models:
#                 outputs = torch.softmax(model(data.to(device)), dim=1)
#                 class_probs.append(outputs)
        
#             pred = torch.stack(class_probs).mean(dim=0)#[:,1].reshape(target.shape)
#             if predictions is None:
#                 predictions = pred
#                 predicted_labels = torch.argmax(pred,dim=1)
#             else:
#                 predictions = torch.cat((predictions,pred),dim=0)   
#                 predicted_labels = torch.cat((predicted_labels,torch.argmax(pred,dim=1)),dim=0) 

#     # Step 2: Extracting the high confidence samples for class 0 and class 1 respectively 
#     class_0_indices = ((predicted_labels == 0).nonzero(as_tuple=False)[:, 0]).detach().cpu().numpy()
#     class_1_indices = ((predicted_labels == 1).nonzero(as_tuple=False)[:, 0]).detach().cpu().numpy()
    
#     print(f'Number of predicted 0s: {len(class_0_indices)}')
#     print(f'Number of predicted 1s: {len(class_1_indices)}')

#     total_samples = X.shape[0]    
#     est_class_1_samples = int(total_samples*avg_CI)    
#     est_class_0_samples = total_samples - est_class_1_samples
#     print(f'Estimated no. of class 0 samples = {est_class_0_samples}')
#     print(f'Estimated no. of class 1 samples = {est_class_1_samples}')
    
#     sorted_pred_class_0 = torch.sort(predictions[class_0_indices, 0], dim=0, descending=True)    
#     top_class_0_indices = sorted_pred_class_0[1][sorted_pred_class_0[0] > 0.8].detach().cpu()
#     if len(top_class_0_indices) > int(labels_ratio*est_class_0_samples):
#         top_class_0_indices = top_class_0_indices[:int(labels_ratio*est_class_0_samples)]
#     print(f'(few) Highest confidence prediction values (class 0): {sorted_pred_class_0[0][:4]}')

#     # print(predictions[class_1_indices, 1])
#     sorted_pred_class_1 = torch.sort(predictions[class_1_indices, 1], dim=0, descending=True)
#     top_class_1_indices = (sorted_pred_class_1[1][sorted_pred_class_1[0] > 0.8]).detach().cpu()
#     if len(top_class_1_indices) > int(labels_ratio*est_class_1_samples):
#         top_class_1_indices = top_class_1_indices[:int(labels_ratio*est_class_1_samples)]
#     print(f'(few) Highest confidence prediction values (class 1): {sorted_pred_class_1[0][:4]}')
    
#     labeled_X,labeled_y,labeled_y_classname = None, None, None
#     X_unlab = None
#     unlabeled_indicies, labeled_indicies = np.array([]), np.array([])
#     member_inference_class_0 = 0 # dummy assignment
#     member_inference_class_1 = 1 # dummy assignment
#     selection_count_0, selection_count_1 = 0, 0
#     n_agreements_0, n_agreements_1 = 0, 0
#     curr_truth_agreement_fraction_0, curr_truth_agreement_fraction_1 = 0, 0

#     # Computing the sample means for each class in memory
#     sample_means = None
#     associated_label = []

#     class_in_memory = np.unique(memory_y_name)
#     print(f'\nclasses in memory: {class_in_memory}')

#     for class_idx in class_in_memory:
#         indices = np.where(memory_y_name == int(class_idx))[0]
#         if str(int(class_idx)) in minorityclass_ids:
#             associated_label.append(1)
#         else:
#             associated_label.append(0)

#         if sample_means is None:
#             sample_means = torch.mean(torch.tensor(memory_X[indices]), dim=0).unsqueeze(0)
#         else:
#             sample_means = torch.cat((sample_means, torch.mean(torch.tensor(memory_X[indices]), dim=0).unsqueeze(0)), dim=0)
#     associated_label = np.array(associated_label)

#     # Setting unique y_classname labels for the new unseen labelled data (for buffer memory storage purpose)
#     if unseen_task:
#         attack_y_name = np.max(class_in_memory) + 1
#         benign_y_name = np.max(class_in_memory) + 2
#         print(f'New classes added to memory: {attack_y_name}, {benign_y_name}')

#     if len(top_class_0_indices) != 0:

#         top_class_0_data = X[class_0_indices[top_class_0_indices]]
#         top_class_0_truth = y[class_0_indices[top_class_0_indices]]
#         top_class_0_y_classname = y_classname[class_0_indices[top_class_0_indices]]

#         # Member inference of the top samples based on distance from buffer memory samples
#         start_inference = time.time()
#         cos_dist = cdist(top_class_0_data, memory_X,'cosine')   
#         # sorted_indices_temp = np.argsort(cos_dist, axis=1)[::-1]
#         # top_k_indices = sorted_indices_temp[:, :1000]
#         # # Create boolean mask with True for top k indices, False otherwise
#         # mask = np.zeros_like(cos_dist, dtype=bool)
#         # mask[np.arange(len(cos_dist))[:, None], top_k_indices] = True
#         # filtered_indices = mask
#         # # filtered_indices = cos_dist <max_value
#         # row_indices_to_keep = np.any(filtered_indices, axis=1)
#         # filtered_arr = filtered_indices[row_indices_to_keep]  
#         # top_class_0_indices = top_class_0_indices[row_indices_to_keep]#removes the indices whose cosine distance > 0.2 
#         # top_class_0_truth = top_class_0_truth[row_indices_to_keep]#removes the indices whose cosine distance > 0.2        
#         # maj_labels = []
#         # for row in filtered_arr:
#         #     maj_labels.append(stats.mode(memory_y[row])[0])
#         # maj_labels = np.array(maj_labels)          
#         # member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
#         # member_inference_class_0 = np.asarray(maj_labels)
#         maj_labels = []
#         row_indices_to_keep = []
#         percentage_mode_value_contributors = []
#         Avg_sample_support = []
#         Avg_sample_support_counter = 0
#         filtered_indices = cos_dist < cos_dist_ip
#         print(top_class_0_data.shape,filtered_indices.shape)
#         row_indices_to_keep = np.where(np.any(filtered_indices, axis=1))[0]
#         rows_to_keep = np.any(filtered_indices, axis=1)
#         for i in range(filtered_indices.shape[0]):
#             valid_indices = np.where(filtered_indices[i])[0]
#             if valid_indices.size > 0:
#                 Avg_sample_support_counter += 1
#                 Avg_sample_support.append(valid_indices.size)
#                 Mode_value_and_count = stats.mode(memory_y[valid_indices])
#                 Mode_value_percentage = (Mode_value_and_count[1]/valid_indices.size)*100
#                 if Mode_value_percentage > mode_value:
#                     maj_labels.append(Mode_value_and_count[0])
#                     percentage_mode_value_contributors.append(Mode_value_percentage)
#                 else:
#                     maj_labels.append(1) 
                
#                 # rows_to_keep.append(True)
#             else:
#                 maj_labels.append(1)#Adding a flipped label for class 0 samples as no confident labels (cost dis <0.2) found in the memory    
#                 # rowrows_to_keeps_to_keep.append(False)
#         print(len(maj_labels))      
#         # exit()  
#         maj_labels = np.array(maj_labels)
#         member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
#         # member_inference_class_0 = np.asarray(maj_labels)
#         print("Average number of sample support for Attack is", stats.tmean(Avg_sample_support), stats.tstd(Avg_sample_support))
#         print("Percentage of samples contributed to each Attack sample is",stats.tmean(percentage_mode_value_contributors),stats.tstd(percentage_mode_value_contributors))
#         # top_class_0_indices = top_class_0_indices[rows_to_keep]#removes the indices whose cosine distance > 0.2 
#         # top_class_0_truth = top_class_0_truth[rows_to_keep]#removes the indices whose cosine distance > 0.2           

       

        
#         end_inference = time.time()
#         print(f'\nNumber of class 0 agreements (between model and member inference): {np.sum(member_inference_class_0 == 0)}/{len(member_inference_class_0)} - ({np.sum(member_inference_class_0 == 0)*100./len(member_inference_class_0):.3f}%)')
#         print(f'Number of class 0 common agreements with ground truth: {np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)}/{np.sum(member_inference_class_0 == 0)} - ({np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)*100./np.sum(member_inference_class_0 == 0):.3f}%)')
#         print(f'Time taken for member inference = {end_inference - start_inference}seconds')

#         n_agreements_0 = np.sum(member_inference_class_0 == 0)
#         curr_truth_agreement_fraction_0 = np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)/np.sum(member_inference_class_0 == 0)
#         if math.isnan(curr_truth_agreement_fraction_0):
#             curr_truth_agreement_fraction_0 = 0

#         # 0-(self)labelled data
#         if unseen_task:
#             if n_agreements_0 > 0:
#                 if truth_agreement_fraction_0 is None or math.isnan(truth_agreement_fraction_0):
#                     truth_agreement_fraction_0 = 1
#                 selection_count_0 = int(n_agreements_0*truth_agreement_fraction_0)
#                 selected_0_indices = np.random.choice(top_class_0_indices[member_inference_class_0 == 0], size=selection_count_0, replace=False)
#                 labeled_indicies = np.hstack((labeled_indicies, selected_0_indices))

#                 labeled_X = np.vstack((labeled_X, X[selected_0_indices])) if labeled_X is not None else X[selected_0_indices]
#                 labeled_y = np.hstack((labeled_y, [0]*selection_count_0)) if labeled_y is not None else [0]*selection_count_0
#                 labeled_y_classname = np.hstack((labeled_y_classname, [benign_y_name]*selection_count_0)) if labeled_y_classname is not None else [benign_y_name]*selection_count_0
#                 print(f'No. of self-labelled samples (class 0): {selection_count_0}')

#                 owl_self_labelled_count_class_0 += selection_count_0
#             else:
#                 print('No. of self-labelled samples (class 0): 0')

#     if len(top_class_1_indices) > 1:#!= 0:
    
#         top_class_1_data = X[class_1_indices[top_class_1_indices]]
#         top_class_1_truth = y[class_1_indices[top_class_1_indices]]
#         top_class_1_y_classname = y_classname[class_1_indices[top_class_1_indices]]
 
#         # Member inference of the top samples based on distance from sample means
#         start_inference = time.time()
#         cos_dist = cdist(top_class_1_data, memory_X,'cosine')   
#         # sorted_indices_temp = np.argsort(cos_dist, axis=1)[::-1]
#         # top_k_indices = sorted_indices_temp[:, :1000]
#         # # Create boolean mask with True for top k indices, False otherwise
#         # mask = np.zeros_like(cos_dist, dtype=bool)
#         # mask[np.arange(len(cos_dist))[:, None], top_k_indices] = True        
#         # filtered_indices = mask
#         # # filtered_indices = cos_dist <max_value
#         # row_indices_to_keep = np.any(filtered_indices, axis=1)
#         # filtered_arr = filtered_indices[row_indices_to_keep]  
#         # top_class_1_indices = top_class_1_indices[row_indices_to_keep]#removes the indices whose cosine distance > 0.2 
#         # top_class_1_truth = top_class_1_truth[row_indices_to_keep]#removes the indices whose cosine distance > 0.2        
#         # maj_labels = []
#         # for row in filtered_arr:
#         #     maj_labels.append(stats.mode(memory_y[row])[0])
#         # maj_labels = np.array(maj_labels)    
#         # maj_labels = []
#         # for row in filtered_arr:
#         #     maj_labels.append(stats.mode(memory_y[row])[0])
#         # maj_labels = np.array(maj_labels)          
#         # member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
#         # member_inference_class_0 = np.asarray(maj_labels)
#         maj_labels = []
#         row_indices_to_keep = []
#         percentage_mode_value_contributors = []
#         Avg_sample_support = []
#         Avg_sample_support_counter = 0
#         filtered_indices = cos_dist < cos_dist_ip
#         print(top_class_1_data.shape,filtered_indices.shape)
#         row_indices_to_keep = np.where(np.any(filtered_indices, axis=1))[0]
#         rows_to_keep = np.any(filtered_indices, axis=1)
#         for i in range(filtered_indices.shape[0]):
#             valid_indices = np.where(filtered_indices[i])[0]
#             if valid_indices.size > 0:
#                 Avg_sample_support_counter += 1
#                 Avg_sample_support.append(valid_indices.size)
#                 Mode_value_and_count = stats.mode(memory_y[valid_indices])
#                 Mode_value_percentage = (Mode_value_and_count[1]/valid_indices.size)*100
#                 if Mode_value_percentage > mode_value:
#                     maj_labels.append(Mode_value_and_count[0])
#                     percentage_mode_value_contributors.append(Mode_value_percentage)
#                 else:
#                     maj_labels.append(0)    

#                 # rows_to_keep.append(True)
#             else:
#                 maj_labels.append(0)#Adding a flipped label for class 0 samples as no confident labels found in the memory    
#             #     rows_to_keep.append(False)
#         print(len(maj_labels))    
#         maj_labels = np.array(maj_labels)
#         member_inference_class_1 = np.asarray(maj_labels.ravel().tolist())
#         print("Average number of sample support for Attack is", stats.tmean(Avg_sample_support), stats.tstd(Avg_sample_support))
#         print("Percentage of samples contributed to each Attack sample is",stats.tmean(percentage_mode_value_contributors),stats.tstd(percentage_mode_value_contributors))
#         # member_inference_class_1 = np.asarray(maj_labels)
#         # top_class_1_indices = top_class_1_indices[rows_to_keep]#removes the indices whose cosine distance > 0.2 
#         # top_class_1_truth = top_class_1_truth[rows_to_keep]#removes the indices whose cosine distance > 0.2 
#         # member_inference_class_1 = np.asarray(maj_labels.ravel().tolist())
#         end_inference = time.time()
#         print(f'\nNumber of class 1 agreements (between model and member inference): {np.sum(member_inference_class_1 == 1)}/{len(member_inference_class_1)} - ({np.sum(member_inference_class_1 == 1)*100./len(member_inference_class_1):.3f})%')
#         print(f'Number of class 1 common agreements with ground truth: {np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)}/{np.sum(member_inference_class_1 == 1)} - ({np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)*100./np.sum(member_inference_class_1 == 1):.3f}%)')
#         print(f'Time taken for member inference = {end_inference - start_inference}seconds')

#         n_agreements_1 = np.sum(member_inference_class_1 == 1)
#         curr_truth_agreement_fraction_1 = np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)/np.sum(member_inference_class_1 == 1)
#         if math.isnan(curr_truth_agreement_fraction_1):
#             curr_truth_agreement_fraction_1 = 0

#         # 1-(self)labelled data
#         if unseen_task:
#             if n_agreements_1 > 0:
#                 if truth_agreement_fraction_1 is None or math.isnan(truth_agreement_fraction_1):
#                     truth_agreement_fraction_1 = 1
                
#                 selection_count_1 = int(n_agreements_1*truth_agreement_fraction_1)
#                 selected_1_indices = np.random.choice(top_class_1_indices[member_inference_class_1 == 1], size=selection_count_1, replace=False)
#                 labeled_indicies = np.hstack((labeled_indicies, selected_1_indices))

#                 labeled_X = np.vstack((labeled_X, X[selected_1_indices])) if labeled_X is not None else X[selected_1_indices]
#                 labeled_y = np.hstack((labeled_y, [1]*selection_count_1)) if labeled_y is not None else [1]*selection_count_1
#                 labeled_y_classname = np.hstack((labeled_y_classname, [attack_y_name]*selection_count_1)) if labeled_y_classname is not None else [attack_y_name]*selection_count_1
                
#                 print(f'No. of self-labelled samples (class 1): {selection_count_1}')
#                 owl_self_labelled_count_class_1 += selection_count_1
#             else:
#                 print('No. of self-labelled samples (class 1): 0')

#     if not unseen_task:
#         return [curr_truth_agreement_fraction_0, curr_truth_agreement_fraction_1]
    
#     print(f'\nTotal no. of self-labeled samples = {selection_count_0 + selection_count_1} (0: {selection_count_0}, 1: {selection_count_1})')

#     # Get security analyst to label the remaining high confidence samples
#     count_class_0 = int(labels_ratio*est_class_0_samples) - selection_count_0 #- n_agreements_0
#     count_class_1 = int(labels_ratio*est_class_1_samples) - selection_count_1 #- n_agreements_1
    
#     remaining_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
#     y_rem = y[remaining_indices]
   
#     remaining_0_indices = remaining_indices[np.where(y_rem == 0)[0]] # remaining indices where y == 0 
#     remaining_1_indices = remaining_indices[np.where(y_rem == 1)[0]] # remaining indices where y == 1
#     print(len(remaining_0_indices), count_class_0)
#     selected_0_indices = np.random.choice(remaining_0_indices, size=min(len(remaining_0_indices), count_class_0), replace=False)
#     selected_1_indices = np.random.choice(remaining_1_indices, size=min(len(remaining_1_indices), count_class_1), replace=False)

#     temp_X = np.vstack((X[selected_0_indices], X[selected_1_indices]))
#     temp_y = np.hstack(([0]*count_class_0, [1]*count_class_1))
#     temp_y_classname = np.hstack(([benign_y_name]*count_class_0, [attack_y_name]*count_class_1))
#     print(f'No. of security analyst-labelled samples: {temp_X.shape[0]} (0:{len(selected_0_indices)}, 1:{len(selected_1_indices)})')

#     owl_analyst_labelled_count_class_0 += len(selected_0_indices)
#     owl_analyst_labelled_count_class_1 += len(selected_1_indices)

#     labeled_X = np.vstack((labeled_X, temp_X)) if labeled_X is not None else temp_X
#     labeled_y = np.hstack((labeled_y, temp_y)) if labeled_y is not None else temp_y
#     labeled_y_classname = np.hstack((labeled_y_classname, temp_y_classname)) if labeled_y_classname is not None else temp_y_classname
#     labeled_indicies = np.hstack((labeled_indicies, np.hstack((selected_0_indices, selected_1_indices))))
#     print(f'Total no. of labelled samples: {labeled_X.shape[0]}')

#     unlabeled_indicies = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
#     X_unlab = X[unlabeled_indicies]
#     y_unlab = y[unlabeled_indicies]
#     y_classname_unlab = y_classname[unlabeled_indicies]
#     print(f'No. of unlabelled samples: {X_unlab.shape}\n')

#     labeled_indicies = labeled_indicies.astype(int)
#     unlabeled_indicies = unlabeled_indicies.astype(int)

#     return labeled_X,labeled_y,labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies
def select_adaptive_percentile(candidate_thresholds):
    """
    Select cosine-distance percentile using only tail behavior.
    Fully empirical and dataset-agnostic.
    """

    ct = np.asarray(candidate_thresholds)

    if len(ct) < 10:
        # Conservative fallback for very small samples
        p = 95
        return p, np.percentile(ct, p)

    # Robust tail statistics
    p50 = np.percentile(ct, 50)
    p99 = np.percentile(ct, 99)

    # Tail heaviness (scale-free)
    tail_ratio = p99 / (p50 + 1e-8)

    # Tail-driven decision
    if tail_ratio > 20:
        chosen_percentile = 99      # very heavy tail
    elif tail_ratio > 10:
        chosen_percentile = 90      # moderately heavy tail
    else:
        chosen_percentile = 25      # light tail

    selected_threshold = np.percentile(ct, chosen_percentile)

    return chosen_percentile, selected_threshold



def owl_data_labeling_strategy(X, y, y_classname, unseen_task=True):

    global owl_self_labelled_count_class_0, owl_self_labelled_count_class_1
    global owl_analyst_labelled_count_class_0, owl_analyst_labelled_count_class_1
    global truth_agreement_fraction_0,truth_agreement_fraction_1
    global avg_CI
    global SELF_LABEL_ACCURACY_LOG
    global task_num
    global adp_attack_cos_dist,adp_benign_cos_dist

    print(f'X shape: {X.shape}')
    dummy_target_label = torch.zeros(X.shape[0])
    data_loader = torch.utils.data.DataLoader(dataset(X,dummy_target_label),
                                            batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                            num_workers=0)
        
    predictions,predicted_labels = None,None

    if mlps == 1:
        models = [student_model1]
    elif mlps == 2:
        models = [student_model1,student_model2]
    else:
        models = [student_model1,student_model2,student_supervised]
    
    for data, _ in data_loader:
        class_probs = [] 
        with torch.no_grad():
            for model in models:
                outputs = torch.softmax(model(data.to(device)), dim=1)
                class_probs.append(outputs)
        
            pred = torch.stack(class_probs).mean(dim=0)#[:,1].reshape(target.shape)
            if predictions is None:
                predictions = pred
                predicted_labels = torch.argmax(pred,dim=1)
            else:
                predictions = torch.cat((predictions,pred),dim=0)   
                predicted_labels = torch.cat((predicted_labels,torch.argmax(pred,dim=1)),dim=0) 

    # Step 2: Extracting the high confidence samples for class 0 and class 1 respectively 
    class_0_indices = ((predicted_labels == 0).nonzero(as_tuple=False)[:, 0]).detach().cpu().numpy()
    class_1_indices = ((predicted_labels == 1).nonzero(as_tuple=False)[:, 0]).detach().cpu().numpy()
    
    print(f'Number of predicted 0s: {len(class_0_indices)}')
    print(f'Number of predicted 1s: {len(class_1_indices)}')

    total_samples = X.shape[0]    
    est_class_1_samples = int(total_samples*avg_CI)    
    est_class_0_samples = total_samples - est_class_1_samples
    print(f'Estimated no. of class 0 samples = {est_class_0_samples}')
    print(f'Estimated no. of class 1 samples = {est_class_1_samples}')
    
    sorted_pred_class_0 = torch.sort(predictions[class_0_indices, 0], dim=0, descending=True)    
    top_class_0_indices = sorted_pred_class_0[1][sorted_pred_class_0[0] > 0.8].detach().cpu()
    if len(top_class_0_indices) > int(labels_ratio*est_class_0_samples):
        top_class_0_indices = top_class_0_indices[:int(labels_ratio*est_class_0_samples)]
    print(f'(few) Highest confidence prediction values (class 0): {sorted_pred_class_0[0][:4]}')

    # print(predictions[class_1_indices, 1])
    sorted_pred_class_1 = torch.sort(predictions[class_1_indices, 1], dim=0, descending=True)
    top_class_1_indices = (sorted_pred_class_1[1][sorted_pred_class_1[0] > 0.8]).detach().cpu()
    if len(top_class_1_indices) > int(labels_ratio*est_class_1_samples):
        top_class_1_indices = top_class_1_indices[:int(labels_ratio*est_class_1_samples)]
    print(f'(few) Highest confidence prediction values (class 1): {sorted_pred_class_1[0][:4]}')
    
    labeled_X,labeled_y,labeled_y_classname = None, None, None
    X_unlab = None
    unlabeled_indicies, labeled_indicies = np.array([]), np.array([])
    member_inference_class_0 = 0 # dummy assignment
    member_inference_class_1 = 1 # dummy assignment
    selection_count_0, selection_count_1 = 0, 0
    n_agreements_0, n_agreements_1 = 0, 0
    curr_truth_agreement_fraction_0, curr_truth_agreement_fraction_1 = 0, 0
    # Exact accuracy of the FINAL self-labeled samples actually used for
    # training this task, vs ground truth -- see SELF_LABEL_ACCURACY_LOG.
    class0_correct, class0_total, class1_correct, class1_total = 0, 0, 0, 0

    # Computing the sample means for each class in memory
    sample_means = None
    associated_label = []

    class_in_memory = np.unique(memory_y_name)
    print(f'\nclasses in memory: {class_in_memory}')

    for class_idx in class_in_memory:
        indices = np.where(memory_y_name == int(class_idx))[0]
        if str(int(class_idx)) in minorityclass_ids:
            associated_label.append(1)
        else:
            associated_label.append(0)

        if sample_means is None:
            sample_means = torch.mean(torch.tensor(memory_X[indices]), dim=0).unsqueeze(0)
        else:
            sample_means = torch.cat((sample_means, torch.mean(torch.tensor(memory_X[indices]), dim=0).unsqueeze(0)), dim=0)
    associated_label = np.array(associated_label)

    # Setting unique y_classname labels for the new unseen labelled data (for buffer memory storage purpose)
    if unseen_task:
        attack_y_name = np.max(class_in_memory) + 1
        benign_y_name = np.max(class_in_memory) + 2
        print(f'New classes added to memory: {attack_y_name}, {benign_y_name}')

    if len(top_class_0_indices) != 0:

        top_class_0_data = X[class_0_indices[top_class_0_indices]]
        top_class_0_truth = y[class_0_indices[top_class_0_indices]]
        top_class_0_y_classname = y_classname[class_0_indices[top_class_0_indices]]

        # Member inference of the top samples based on distance from buffer memory samples
        start_inference = time.time()
        cos_dist = cdist(top_class_0_data, memory_X,'cosine')  
        
        if unseen_task == False:
            # ================= Estimate cos_dist_ adaptively for benign =================
            memory_y_np = np.asarray(memory_y) 
            candidate_thresholds = []
            cos_time = time.time()
            for i in range(cos_dist.shape[0]):
                # sort memory samples by cosine distance
                sorted_idx = np.argsort(cos_dist[i])
                sorted_dist = cos_dist[i][sorted_idx]
                # sorted_labels = memory_y[sorted_idx]
                sorted_labels = memory_y_np[sorted_idx]

                # progressively expand neighborhood
                benign_count = 0
                attack_count = 0
                amount_of_samples = int(0.05*len(sorted_labels))
                loop_range = amount_of_samples
                for k in range(0, len(sorted_labels), loop_range):
                    chunk = sorted_labels[k:k + loop_range]

                    # count labels in this chunk
                    benign_count += np.sum(chunk == 0)
                    attack_count += np.sum(chunk == 1)

                    total = benign_count + attack_count
                    if total == 0:
                        continue

                    # compute benign agreement
                    mode_percentage = (max(benign_count, attack_count) / total) * 100
                    mode_label = 0 if benign_count >= attack_count else 1

                    if mode_label == 0 and mode_percentage >= mode_value:
                        candidate_thresholds.append(sorted_dist[min(k + loop_range-1, len(sorted_dist) - 1)])
                        break
                # for k in range(5, len(sorted_dist), 500):  # small neighborhoods first
                #     neighbor_labels = sorted_labels[:k]
                #     mode_label, mode_count = stats.mode(neighbor_labels)
                #     mode_percentage = (mode_count / k) * 100

                #     # accept only if benign agreement is strong
                #     if mode_label == 0 and mode_percentage >= mode_value:
                #         candidate_thresholds.append(sorted_dist[k - 1])
                #         break
                    

            # robust aggregation
            # print(candidate_thresholds)
#             print(
#     f"candidate_thresholds | "
#     f"min: {min(candidate_thresholds):.4f}, "
#     f"max: {max(candidate_thresholds):.4f}, "
#     f"avg: {sum(candidate_thresholds)/len(candidate_thresholds):.4f},"
#     f"25 percentile: {np.percentile(candidate_thresholds, 25):.4f},"
#     f"50 percentile: {np.percentile(candidate_thresholds, 50):.4f},"
#     f"75 percentile: {np.percentile(candidate_thresholds, 75):.4f},"
#     f"90 percentile: {np.percentile(candidate_thresholds, 90):.4f},"
#     f"95 percentile: {np.percentile(candidate_thresholds, 95):.4f},"
#     f"98 percentile: {np.percentile(candidate_thresholds, 98):.4f},"
#     f"99 percentile: {np.percentile(candidate_thresholds, 99):.4f},"
# )

            # exit()
            if len(candidate_thresholds) > 0:
                if adp_benign_cos_dist > 0:
                    # adp_benign_cos_dist = 0.9 * adp_benign_cos_dist + 0.1 * np.percentile(candidate_thresholds, 99)
                    # adp_benign_cos_dist = 0.9 * adp_benign_cos_dist + 0.1 * select_adaptive_percentile(candidate_thresholds)[1]
                    adp_benign_cos_dist = 0.1 * adp_benign_cos_dist + 0.9 * select_adaptive_percentile(candidate_thresholds)[1]
                                   
                else:
                    # adp_benign_cos_dist = np.percentile(candidate_thresholds, 99)
                    adp_benign_cos_dist = select_adaptive_percentile(candidate_thresholds)[1]
                # cos_dist_ip = np.percentile(candidate_thresholds, 75)
            else:
                adp_benign_cos_dist = 0.2
                # cos_dist_ip = 0.2  # safe fallback

            time_elapsed = time.time()-cos_time
            print(f"[TIMER] Task-level cos_dist_ip computation time: {time_elapsed:.2f} seconds")
            print(f"[INFO] Adaptive cosine distance threshold = {adp_benign_cos_dist:.4f}")
        # exit()

        # sorted_indices_temp = np.argsort(cos_dist, axis=1)[::-1]
        # top_k_indices = sorted_indices_temp[:, :1000]
        # # Create boolean mask with True for top k indices, False otherwise
        # mask = np.zeros_like(cos_dist, dtype=bool)
        # mask[np.arange(len(cos_dist))[:, None], top_k_indices] = True
        # filtered_indices = mask
        # # filtered_indices = cos_dist <max_value
        # row_indices_to_keep = np.any(filtered_indices, axis=1)
        # filtered_arr = filtered_indices[row_indices_to_keep]  
        # top_class_0_indices = top_class_0_indices[row_indices_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_0_truth = top_class_0_truth[row_indices_to_keep]#removes the indices whose cosine distance > 0.2        
        # maj_labels = []
        # for row in filtered_arr:
        #     maj_labels.append(stats.mode(memory_y[row])[0])
        # maj_labels = np.array(maj_labels)          
        # member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
        # member_inference_class_0 = np.asarray(maj_labels)
        maj_labels = []
        row_indices_to_keep = []
        percentage_mode_value_contributors = []
        Avg_sample_support = []
        Avg_sample_support_counter = 0
        # filtered_indices = cos_dist < cos_dist_ip
        filtered_indices = cos_dist < adp_benign_cos_dist
        print(top_class_0_data.shape,filtered_indices.shape)
        row_indices_to_keep = np.where(np.any(filtered_indices, axis=1))[0]
        rows_to_keep = np.any(filtered_indices, axis=1)
        for i in range(filtered_indices.shape[0]):
            valid_indices = np.where(filtered_indices[i])[0]
            if valid_indices.size > 0:
                Avg_sample_support_counter += 1
                Avg_sample_support.append(valid_indices.size)
                Mode_value_and_count = stats.mode(memory_y[valid_indices])
                Mode_value_percentage = (Mode_value_and_count[1]/valid_indices.size)*100
                if Mode_value_percentage > mode_value:
                    maj_labels.append(Mode_value_and_count[0])
                    percentage_mode_value_contributors.append(Mode_value_percentage)
                else:
                    maj_labels.append(1) 
                
                # rows_to_keep.append(True)
            else:
                maj_labels.append(1)#Adding a flipped label for class 0 samples as no confident labels (cost dis <0.2) found in the memory    
                # rowrows_to_keeps_to_keep.append(False)
        print(len(maj_labels))      
        # exit()  
        maj_labels = np.array(maj_labels)
        member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
        # member_inference_class_0 = np.asarray(maj_labels)
        print("Average number of sample support for Attack is", stats.tmean(Avg_sample_support), stats.tstd(Avg_sample_support))
        print("Percentage of samples contributed to each Attack sample is",stats.tmean(percentage_mode_value_contributors),stats.tstd(percentage_mode_value_contributors))
        # top_class_0_indices = top_class_0_indices[rows_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_0_truth = top_class_0_truth[rows_to_keep]#removes the indices whose cosine distance > 0.2           

       

        
        end_inference = time.time()
        print(f'\nNumber of class 0 agreements (between model and member inference): {np.sum(member_inference_class_0 == 0)}/{len(member_inference_class_0)} - ({np.sum(member_inference_class_0 == 0)*100./len(member_inference_class_0):.3f}%)')
        print(f'Number of class 0 common agreements with ground truth: {np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)}/{np.sum(member_inference_class_0 == 0)} - ({np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)*100./np.sum(member_inference_class_0 == 0):.3f}%)')
        print(f'Time taken for member inference = {end_inference - start_inference}seconds')

        n_agreements_0 = np.sum(member_inference_class_0 == 0)
        curr_truth_agreement_fraction_0 = np.sum(top_class_0_truth[member_inference_class_0 == 0] == 0)/np.sum(member_inference_class_0 == 0)
        if math.isnan(curr_truth_agreement_fraction_0):
            curr_truth_agreement_fraction_0 = 0

        # 0-(self)labelled data
        if unseen_task:
            if n_agreements_0 > 0:
                if truth_agreement_fraction_0 is None or math.isnan(truth_agreement_fraction_0):
                    truth_agreement_fraction_0 = 1
                selection_count_0 = int(n_agreements_0*truth_agreement_fraction_0)
                selected_0_indices = np.random.choice(top_class_0_indices[member_inference_class_0 == 0], size=selection_count_0, replace=False)

                # SELF-LABEL ACCURACY (class 0/benign): exact ground-truth check on
                # the samples actually chosen as self-labeled here -- measurement
                # only, y is never used to influence the selection or training.
                class0_total = selection_count_0
                class0_correct = int(np.sum(y[class_0_indices[selected_0_indices]] == 0))

                labeled_indicies = np.hstack((labeled_indicies, selected_0_indices))

                labeled_X = np.vstack((labeled_X, X[selected_0_indices])) if labeled_X is not None else X[selected_0_indices]
                labeled_y = np.hstack((labeled_y, [0]*selection_count_0)) if labeled_y is not None else [0]*selection_count_0
                labeled_y_classname = np.hstack((labeled_y_classname, [benign_y_name]*selection_count_0)) if labeled_y_classname is not None else [benign_y_name]*selection_count_0
                print(f'No. of self-labelled samples (class 0): {selection_count_0}')

                owl_self_labelled_count_class_0 += selection_count_0
            else:
                print('No. of self-labelled samples (class 0): 0')

    if len(top_class_1_indices) > 1:#!= 0:
    
        top_class_1_data = X[class_1_indices[top_class_1_indices]]
        top_class_1_truth = y[class_1_indices[top_class_1_indices]]
        top_class_1_y_classname = y_classname[class_1_indices[top_class_1_indices]]
 
        # Member inference of the top samples based on distance from sample means
        start_inference = time.time()
        cos_dist = cdist(top_class_1_data, memory_X,'cosine')   
        if unseen_task == False:

            # ================= Estimate cos_dist_ adaptively for attack =================
            memory_y_np = np.asarray(memory_y)
            candidate_thresholds_attack = []

            cos_time = time.time()

            for i in range(cos_dist.shape[0]):

                # sort memory samples by cosine distance
                sorted_idx = np.argsort(cos_dist[i])
                sorted_dist = cos_dist[i][sorted_idx]
                sorted_labels = memory_y_np[sorted_idx]

                benign_count = 0
                attack_count = 0
                amount_of_samples = int(0.05*len(sorted_labels))
                loop_range = amount_of_samples  # chunk size

                for k in range(0, len(sorted_labels), loop_range):
                    chunk = sorted_labels[k:k + loop_range]

                    # count labels in this chunk
                    benign_count += np.sum(chunk == 0)
                    attack_count += np.sum(chunk == 1)

                    total = benign_count + attack_count
                    if total == 0:
                        continue

                    # compute attack agreement
                    mode_percentage = (max(benign_count, attack_count) / total) * 100
                    mode_label = 1 if attack_count >= benign_count else 0

                    # accept only if ATTACK agreement is strong
                    if mode_label == 1 and mode_percentage >= mode_value:
                        candidate_thresholds_attack.append(
                            sorted_dist[min(k + loop_range - 1, len(sorted_dist) - 1)]
                        )
                        break
#             print(
#     f"candidate_thresholds | "
#     f"min: {min(candidate_thresholds_attack):.4f}, "
#     f"max: {max(candidate_thresholds_attack):.4f}, "
#     f"avg: {sum(candidate_thresholds_attack)/len(candidate_thresholds_attack):.4f},"
#     f"25 percentile: {np.percentile(candidate_thresholds_attack, 25):.4f},"
#     f"50 percentile: {np.percentile(candidate_thresholds_attack, 50):.4f},"
#     f"75 percentile: {np.percentile(candidate_thresholds_attack, 75):.4f},"
#     f"90 percentile: {np.percentile(candidate_thresholds_attack, 90):.4f},"
#     f"95 percentile: {np.percentile(candidate_thresholds_attack, 95):.4f},"
#     f"98 percentile: {np.percentile(candidate_thresholds_attack, 98):.4f},"
#     f"99 percentile: {np.percentile(candidate_thresholds_attack, 99):.4f},"
# )
            # exit()
            # robust aggregation
            if len(candidate_thresholds_attack) > 0:
                if adp_attack_cos_dist > 0:
                    # adp_attack_cos_dist = 0.9*adp_attack_cos_dist+0.1*np.percentile(candidate_thresholds_attack, 99)
                    # adp_attack_cos_dist = 0.9*adp_attack_cos_dist+0.1*select_adaptive_percentile(candidate_thresholds_attack)[1]
                    adp_attack_cos_dist = 0.1*adp_attack_cos_dist+0.9*select_adaptive_percentile(candidate_thresholds_attack)[1]
                else:
                    # adp_attack_cos_dist = np.percentile(candidate_thresholds_attack, 99)
                    adp_attack_cos_dist = select_adaptive_percentile(candidate_thresholds_attack)[1]
            else:
                adp_attack_cos_dist = 0.2  # safe fallback

            time_elapsed = time.time() - cos_time
            print(f"[TIMER] Adaptive ATTACK cosine distance estimation: {time_elapsed:.3f}s")
            print(f"[INFO] Adaptive cosine distance threshold = {adp_attack_cos_dist:.4f}")
            # exit()

        # sorted_indices_temp = np.argsort(cos_dist, axis=1)[::-1]
        # top_k_indices = sorted_indices_temp[:, :1000]
        # # Create boolean mask with True for top k indices, False otherwise
        # mask = np.zeros_like(cos_dist, dtype=bool)
        # mask[np.arange(len(cos_dist))[:, None], top_k_indices] = True        
        # filtered_indices = mask
        # # filtered_indices = cos_dist <max_value
        # row_indices_to_keep = np.any(filtered_indices, axis=1)
        # filtered_arr = filtered_indices[row_indices_to_keep]  
        # top_class_1_indices = top_class_1_indices[row_indices_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_1_truth = top_class_1_truth[row_indices_to_keep]#removes the indices whose cosine distance > 0.2        
        # maj_labels = []
        # for row in filtered_arr:
        #     maj_labels.append(stats.mode(memory_y[row])[0])
        # maj_labels = np.array(maj_labels)    
        # maj_labels = []
        # for row in filtered_arr:
        #     maj_labels.append(stats.mode(memory_y[row])[0])
        # maj_labels = np.array(maj_labels)          
        # member_inference_class_0 = np.asarray(maj_labels.ravel().tolist())
        # member_inference_class_0 = np.asarray(maj_labels)
        maj_labels = []
        row_indices_to_keep = []
        percentage_mode_value_contributors = []
        Avg_sample_support = []
        Avg_sample_support_counter = 0
        # filtered_indices = cos_dist < cos_dist_ip
        filtered_indices = cos_dist < adp_attack_cos_dist
        print(top_class_1_data.shape,filtered_indices.shape)
        row_indices_to_keep = np.where(np.any(filtered_indices, axis=1))[0]
        rows_to_keep = np.any(filtered_indices, axis=1)
        for i in range(filtered_indices.shape[0]):
            valid_indices = np.where(filtered_indices[i])[0]
            if valid_indices.size > 0:
                Avg_sample_support_counter += 1
                Avg_sample_support.append(valid_indices.size)
                Mode_value_and_count = stats.mode(memory_y[valid_indices])
                Mode_value_percentage = (Mode_value_and_count[1]/valid_indices.size)*100
                if Mode_value_percentage > mode_value:
                    maj_labels.append(Mode_value_and_count[0])
                    percentage_mode_value_contributors.append(Mode_value_percentage)
                else:
                    maj_labels.append(0)    

                # rows_to_keep.append(True)
            else:
                maj_labels.append(0)#Adding a flipped label for class 0 samples as no confident labels found in the memory    
            #     rows_to_keep.append(False)
        print(len(maj_labels))    
        maj_labels = np.array(maj_labels)
        member_inference_class_1 = np.asarray(maj_labels.ravel().tolist())
        print("Average number of sample support for Attack is", stats.tmean(Avg_sample_support), stats.tstd(Avg_sample_support))
        print("Percentage of samples contributed to each Attack sample is",stats.tmean(percentage_mode_value_contributors),stats.tstd(percentage_mode_value_contributors))
        # member_inference_class_1 = np.asarray(maj_labels)
        # top_class_1_indices = top_class_1_indices[rows_to_keep]#removes the indices whose cosine distance > 0.2 
        # top_class_1_truth = top_class_1_truth[rows_to_keep]#removes the indices whose cosine distance > 0.2 
        # member_inference_class_1 = np.asarray(maj_labels.ravel().tolist())
        end_inference = time.time()
        print(f'\nNumber of class 1 agreements (between model and member inference): {np.sum(member_inference_class_1 == 1)}/{len(member_inference_class_1)} - ({np.sum(member_inference_class_1 == 1)*100./len(member_inference_class_1):.3f})%')
        print(f'Number of class 1 common agreements with ground truth: {np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)}/{np.sum(member_inference_class_1 == 1)} - ({np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)*100./np.sum(member_inference_class_1 == 1):.3f}%)')
        print(f'Time taken for member inference = {end_inference - start_inference}seconds')

        n_agreements_1 = np.sum(member_inference_class_1 == 1)
        curr_truth_agreement_fraction_1 = np.sum(top_class_1_truth[member_inference_class_1 == 1] == 1)/np.sum(member_inference_class_1 == 1)
        if math.isnan(curr_truth_agreement_fraction_1):
            curr_truth_agreement_fraction_1 = 0

        # 1-(self)labelled data
        if unseen_task:
            if n_agreements_1 > 0:
                if truth_agreement_fraction_1 is None or math.isnan(truth_agreement_fraction_1):
                    truth_agreement_fraction_1 = 1
                
                selection_count_1 = int(n_agreements_1*truth_agreement_fraction_1)
                selected_1_indices = np.random.choice(top_class_1_indices[member_inference_class_1 == 1], size=selection_count_1, replace=False)

                # SELF-LABEL ACCURACY (class 1/attack): same measurement as class 0
                # above, exact ground-truth check on the samples actually selected.
                class1_total = selection_count_1
                class1_correct = int(np.sum(y[class_1_indices[selected_1_indices]] == 1))

                labeled_indicies = np.hstack((labeled_indicies, selected_1_indices))

                labeled_X = np.vstack((labeled_X, X[selected_1_indices])) if labeled_X is not None else X[selected_1_indices]
                labeled_y = np.hstack((labeled_y, [1]*selection_count_1)) if labeled_y is not None else [1]*selection_count_1
                labeled_y_classname = np.hstack((labeled_y_classname, [attack_y_name]*selection_count_1)) if labeled_y_classname is not None else [attack_y_name]*selection_count_1
                
                print(f'No. of self-labelled samples (class 1): {selection_count_1}')
                owl_self_labelled_count_class_1 += selection_count_1
            else:
                print('No. of self-labelled samples (class 1): 0')

    if not unseen_task:
        return [curr_truth_agreement_fraction_0, curr_truth_agreement_fraction_1]

    # Log this task's self-label accuracy (class0=benign, class1=attack), exact
    # counts against ground truth on the samples actually used for training.
    c0_acc = class0_correct / class0_total if class0_total > 0 else float('nan')
    c1_acc = class1_correct / class1_total if class1_total > 0 else float('nan')
    combined_total = class0_total + class1_total
    combined_acc = (class0_correct + class1_correct) / combined_total if combined_total > 0 else float('nan')
    SELF_LABEL_ACCURACY_LOG.append({
        'task_id': task_num,
        'class0_correct': class0_correct, 'class0_total': class0_total,
        'class1_correct': class1_correct, 'class1_total': class1_total,
    })
    print(f'[SELF-LABEL-ACC] task={task_num}  '
          f'class0(benign)={class0_correct}/{class0_total} ({c0_acc*100:.2f}%)  '
          f'class1(attack)={class1_correct}/{class1_total} ({c1_acc*100:.2f}%)  '
          f'combined={class0_correct+class1_correct}/{combined_total} ({combined_acc*100:.2f}%)')
    
    print(f'\nTotal no. of self-labeled samples = {selection_count_0 + selection_count_1} (0: {selection_count_0}, 1: {selection_count_1})')

    # Get security analyst to label the remaining high confidence samples
    count_class_0 = int(labels_ratio*est_class_0_samples) - selection_count_0 #- n_agreements_0
    count_class_1 = int(labels_ratio*est_class_1_samples) - selection_count_1 #- n_agreements_1
    
    remaining_indices = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    y_rem = y[remaining_indices]
   
    remaining_0_indices = remaining_indices[np.where(y_rem == 0)[0]] # remaining indices where y == 0 
    remaining_1_indices = remaining_indices[np.where(y_rem == 1)[0]] # remaining indices where y == 1
    print(len(remaining_0_indices), count_class_0)
    selected_0_indices = np.random.choice(remaining_0_indices, size=min(len(remaining_0_indices), count_class_0), replace=False)
    selected_1_indices = np.random.choice(remaining_1_indices, size=min(len(remaining_1_indices), count_class_1), replace=False)

    temp_X = np.vstack((X[selected_0_indices], X[selected_1_indices]))
    temp_y = np.hstack(([0]*count_class_0, [1]*count_class_1))
    temp_y_classname = np.hstack(([benign_y_name]*count_class_0, [attack_y_name]*count_class_1))
    print(f'No. of security analyst-labelled samples: {temp_X.shape[0]} (0:{len(selected_0_indices)}, 1:{len(selected_1_indices)})')

    owl_analyst_labelled_count_class_0 += len(selected_0_indices)
    owl_analyst_labelled_count_class_1 += len(selected_1_indices)

    labeled_X = np.vstack((labeled_X, temp_X)) if labeled_X is not None else temp_X
    labeled_y = np.hstack((labeled_y, temp_y)) if labeled_y is not None else temp_y
    labeled_y_classname = np.hstack((labeled_y_classname, temp_y_classname)) if labeled_y_classname is not None else temp_y_classname
    labeled_indicies = np.hstack((labeled_indicies, np.hstack((selected_0_indices, selected_1_indices))))
    print(f'Total no. of labelled samples: {labeled_X.shape[0]}')

    unlabeled_indicies = np.setdiff1d(np.arange(X.shape[0]), labeled_indicies)
    X_unlab = X[unlabeled_indicies]
    y_unlab = y[unlabeled_indicies]
    y_classname_unlab = y_classname[unlabeled_indicies]
    print(f'No. of unlabelled samples: {X_unlab.shape}\n')

    labeled_indicies = labeled_indicies.astype(int)
    unlabeled_indicies = unlabeled_indicies.astype(int)

    return labeled_X,labeled_y,labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies


# def train(str_train_model,tasks,task_class_ids,task_id,feature_list,threshold,X_val,y_val,bool_reorganize_memory,owl_data_labeling=False):
    
#     global memory_X, memory_y, memory_y_name,local_count,global_count,local_store,input_shape,memory_size,task_num
#     global classes_so_far,full,global_priority_list,local_priority_list,memory_population_time,replay_size
#     global memory_population_time,epochs,grad_norm_dict,temp_norm
#     global student_optimizer1,student_optimizer2,student_supervised_optimizer
#     global teacher_model1,teacher_model2,teacher_supervised,student_model1,student_model2,student_supervised
#     global truth_agreement_fraction_0, truth_agreement_fraction_1 
#     global avg_CI, CI_list 

#     if str_train_model == "student1":
#         model = student_model1
#         opt = student_optimizer1
#         teacher_model = teacher_model1
#     elif str_train_model == "student2":
#         model = student_model2
#         opt = student_optimizer2
#         teacher_model = teacher_model2
#     elif str_train_model == "student_supervised":
#         model = student_supervised
#         opt = student_supervised_optimizer    
#         teacher_model = teacher_supervised    

#     grad_norm_list = []

#     valid_loader = torch.utils.data.DataLoader(dataset(X_val,y_val),
#                                                batch_size=batch_size,
#                                             #    sampler=valid_sampler,
#                                                num_workers=0)
#     feature_mat = []
#     X,y,y_classname = tasks[0][0],tasks[0][1],tasks[0][2]
#     y_large,y_small = max(np.sum(y == 0),np.sum(y == 1)),min(np.sum(y == 0),np.sum(y == 1))
#     print("majority class",y_large)
#     print("minority class",y_small)
#     print("class imbalance ratio",y_small/(y_large+y_small))
#     unique_y_classname = np.unique(y_classname)
#     if unique_y_classname[0]%2 == 0:
#         attack_y_name = unique_y_classname[0]
#         benign_y_name = unique_y_classname[1]
#     else:
#         attack_y_name = unique_y_classname[1]
#         benign_y_name = unique_y_classname[0]

#     # if task_id > 0:
#     #     compute_otdd(task_id, X, memory_X, memory_y_name, attack_y_name, benign_y_name)

#     task_size = X.shape[0]
#     if owl_data_labeling == False:

#         if task_id == 0:
#             labeled_indicies,unlabeled_indicies=split_a_task(tasks,0.99,task_class_ids)
#         else:
#             labeled_indicies,unlabeled_indicies=split_a_task(tasks,labels_ratio,task_class_ids)
#         labeled_X,labeled_y,labeled_y_classname = X[labeled_indicies],y[labeled_indicies],y_classname[labeled_indicies]
#         X_unlab,y_unlab,y_unlabclassname = X[unlabeled_indicies],y[unlabeled_indicies],y_classname[unlabeled_indicies]
        
#         # Computing class imbalance in the labeled samples
#         maj_class_count,min_class_count = max(np.sum(labeled_y == 0),np.sum(labeled_y == 1)),min(np.sum(labeled_y == 0),np.sum(labeled_y == 1))
#         CI_list.append(min_class_count/(maj_class_count+min_class_count))
#         # CI_list.append(maj_class_count/(min_class_count))
#         # CI_list.append(np.sum(labeled_y == 0)/(np.sum(labeled_y == 1) + np.sum(labeled_y == 0)))
#         # print("majority samples",maj_class_count)
#         # print("minority samples",min_class_count)
#         # CI_list.append(np.sum(labeled_y == 0)/(np.sum(labeled_y == 1) + np.sum(labeled_y == 0)))
#         print(f'Class Imbalance for task {task_id} (% of class 0 samples)= {CI_list[-1]}\n')
#         avg_CI = np.mean(CI_list)

#         # Computing class imbalance ratio for the task (0:1)
#         if task_id > 0:
#             task_truth_agreement_fractions = owl_data_labeling_strategy(X, y, y_classname, unseen_task=False)
#             print(f'\nCurrent task truth agreement fractions = {task_truth_agreement_fractions}')
#             print(truth_agreement_fraction_0, truth_agreement_fraction_1)
#             ## accumulation
#             if truth_agreement_fraction_0 is None:
#                 truth_agreement_fraction_0 = task_truth_agreement_fractions[0] if task_truth_agreement_fractions[0] != 0 else None
#             else:
#                 truth_agreement_fraction_0 = beta*truth_agreement_fraction_0 + (1 - beta)*task_truth_agreement_fractions[0] if task_truth_agreement_fractions[0] != 0 else truth_agreement_fraction_0
            
#             if truth_agreement_fraction_1 is None:
#                 truth_agreement_fraction_1 = task_truth_agreement_fractions[1] if task_truth_agreement_fractions[1] != 0 else None
#             else:
#                 truth_agreement_fraction_1 = beta*truth_agreement_fraction_1 + (1 - beta)*task_truth_agreement_fractions[1] if task_truth_agreement_fractions[1] != 0 else truth_agreement_fraction_1

#             # truth_agreement_fraction_0, truth_agreement_fraction_1 = min(0.5,truth_agreement_fraction_0),min(0.5,truth_agreement_fraction_1)
#             print(truth_agreement_fraction_0, truth_agreement_fraction_1)
#         #     print()


#     else:
#         labeled_X, labeled_y, labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies = owl_data_labeling_strategy(X, y, y_classname, unseen_task=True)
        
#         # labeled_X_class_0,labeled_X_class_1,X_unlab = open_world_data_labeling(X)
#         # labeled_X = np.concatenate((labeled_X_class_0,labeled_X_class_1),axis=0)
#         # labeled_y = np.concatenate((np.zeros((labeled_X_class_0.shape[0],), dtype=np.int64),np.ones((labeled_X_class_1.shape[0],), dtype=np.int64)),axis=0)
#         # labeled_y_classname = np.concatenate((np.full((labeled_X_class_0.shape[0],),fill_value=benign_y_name,dtype=np.int64),np.full((labeled_X_class_1.shape[0],),fill_value=attack_y_name,dtype=np.int64)),axis=0)
#         # random_indices = np.random.permutation(len(labeled_X))
#         # labeled_X,labeled_y,labeled_y_classname = X[random_indices],y[random_indices],y_classname[random_indices]
#         # labeled_indicies = [lab_idx_num for lab_idx_num in range(labeled_X.shape[0])]
#         # unlabeled_indicies = [unlab_idx_num+len(labeled_indicies) for unlab_idx_num in range(X_unlab.shape[0])]
#         # labeled_indicies,unlabeled_indicies=open_world_data_labeling(X)
#         # labeled_X,labeled_y,labeled_y_classname = X[labeled_indicies],y[labeled_indicies],y_classname[labeled_indicies]
#         # X_unlab,y_unlab,y_unlabclassname = X[unlabeled_indicies],y[unlabeled_indicies],y_classname[unlabeled_indicies]
        
#         # labeled_X_class_0,labeled_X_class_1,X_unlab = open_world_data_labeling(X)
#         # print("expected class zero labels",labeled_X_class_0.shape)
#         # print("actual class zero labels,",Counter(y[labeled_X_class_0]))
#         # print("top 20 class 0 labels",y[labeled_X_class_0[0:20]])
#         # print("expected class one labels",labeled_X_class_1.shape)
#         # print("actual class one labels,",Counter(y[labeled_X_class_1]))
    
    
#     if task_id > 0:
              
#             mem_batch_size = floor(batch_size*b_m)
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
#         # labeled_batch_size = floor(batch_size*0.99)
#         labeled_batch_size = floor(batch_size*labels_ratio)
#         unlabeled_batch_size = batch_size-labeled_batch_size
#         no_of_batches = floor(task_size/batch_size)

#     if bool_gpm:
#         for i in range(len(feature_list)):
#             Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
#             feature_mat.append(Uf)    
    

#     ###Buffer memory organization
#     temp_x,temp_y,temp_yname = labeled_X,labeled_y,labeled_y_classname
#     if task_id > 0 and bool_reorganize_memory:
#         mem_start_time = time.time()
#         if str(mem_strat) == "replace":
            
#             tasks[0] = temp_x,temp_y,temp_yname
#             lab_samples_in_memory = split_a_task(tasks,lab_samp_in_mem_ratio)
#             tasks[0] = temp_x[lab_samples_in_memory[0],:],temp_y[lab_samples_in_memory[0]],temp_yname[lab_samples_in_memory[0]]
#             initialize_buffermemory(tasks=tasks,mem_size=memory_size)
#         elif str(mem_strat) == "equal":
            
#             memory_X, memory_y, memory_y_name = memory_update_equal_allocation2(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.15,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)
#         else:
            
#             memory_X, memory_y, memory_y_name = memory_update_equal_allocation(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.85,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)

#         mem_finish_time = time.time()
#         memory_population_time += mem_finish_time-mem_start_time
#     # prog_bar = tqdm(range(no_of_batches))
#     # for batch_idx in prog_bar:
#     # to track the training loss as the model trains
#     train_losses = []
#     # to track the validation loss as the model trains
#     valid_losses = []
#     # to track the average training loss per epoch as the model trains
#     avg_train_losses = []
#     # to track the average validation loss per epoch as the model trains
#     avg_valid_losses = [] 
#     check_point_file_name = "checkpoint"+str(os.getpid())+".pt"
#     check_point_file_name_norm = "checkpoint"+str(os.getpid())+"grad_norm"+".pt"
#     early_stopping = EarlyStopping(patience=3, verbose=True,path=check_point_file_name)
#     gradient_rejection = GradientRejection(patience=2, verbose=True,path=check_point_file_name_norm)
#     scheduler = StepLR(opt, step_size=1, gamma=0.96)
#     for epoch in range(epochs):
#         # print("epoch",epoch)
#         # scheduler.step()
#         prog_bar = tqdm(range(no_of_batches))
#         for batch_idx in prog_bar:
#             model.train()        
#         # for epoch in range(epochs):
#             with torch.no_grad():
#                 if task_id > 0 and batch_idx < no_of_unlab_batches:
#                     unlabeled_X = torch.from_numpy(X_unlab[batch_idx*unlabeled_batch_size:batch_idx*unlabeled_batch_size+unlabeled_batch_size]).to(device)
#                 else:
#                     rand_indices = list(random.sample(range(X_unlab.shape[0]),min(unlabeled_batch_size,X_unlab.shape[0])))
#                     unlabeled_X = torch.from_numpy(X_unlab[rand_indices]).to(device)


#                 if image_resolution is not None:
#                     unlabeled_X = unlabeled_X.reshape(image_resolution)
#                 unlabeled_pred = torch.softmax(model(unlabeled_X),dim=1).detach()
            
#             lab_X = labeled_X[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]  
#             lab_y = labeled_y[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]
#             task_lab_X,task_lab_y = lab_X,lab_y
#             if task_id > 0:
                
#                 mem_batch = sample_batch_from_memory(floor(batch_size*b_m),minority_alloc=batch_minority_alloc)
#                 if mem_batch is not None and mem_batch[0].shape[0] > 0:
                    
#                     lab_X = np.concatenate((lab_X,mem_batch[0]), axis=0)  
#                     # temp_mem_X = torch.from_numpy(mem_batch[0]).to(device)
#                     # if image_resolution is not None:
#                     #     temp_mem_X = temp_mem_X.reshape(image_resolution)
#                     # temp_mem_y = torch.argmax(teacher_model(temp_mem_X),dim=1).detach().cpu().numpy().squeeze()
                    
#                     # lab_y = np.concatenate((lab_y,temp_mem_y), axis=0)
                    
#                     lab_y = np.concatenate((lab_y,mem_batch[1]), axis=0)
#             lab_X = torch.from_numpy(lab_X).to(device)
#             if image_resolution is not None:
#                     lab_X = lab_X.reshape(image_resolution)
                   
#             # print(model(lab_X))
#             y_pred = torch.softmax(model(lab_X),dim=1).squeeze()#.to(device)                      
#             lab_y = torch.from_numpy(lab_y).to(device).to(dtype=torch.long)#.reshape(y_pred.shape)

#             # lab_y = F.one_hot(lab_y, 2)
#             sup_loss = loss_fn(y_pred.float(),F.one_hot(lab_y.to(dtype=torch.long), 2).float())#to(device)
#             # sup_loss = loss_fn(y_pred,lab_y.float())
#             total_loss = sup_loss
            
#             distil_loss = 0
#             distil_loss = torch.as_tensor(distil_loss).to(device)
#             opt.zero_grad()
#             if task_id > 0 and train_with_unlab:
#                 if str_train_model!="student_supervised":
#                     #computing the distillation loss
#                     distil_loss_list = compute_distill_loss(unlabeled_pred,unlabeled_X)
#                     distil_loss = distil_loss_list[0]
#                     total_loss = total_loss +  alpha *distil_loss  
#                     # lab_X = torch.cat((lab_X,unlabeled_X),0)
#                     # lab_y = torch.cat((lab_y,distil_loss_list[1]),0)

#                 contrast_loss = 0
                
#                 if bool_closs:
#                     # positives, negatives = construct_positive_negative_samples(lab_X, lab_y)  
#                     positives, negatives = construct_positive_negative_samples_from_memory(task_lab_y)  
#                     anchor_representations = model(torch.from_numpy(task_lab_X).to(device))
#                     positive_representations = model(positives)
#                     negative_representations = model(negatives)
#                     contrast_loss = contrastive_loss(anchor_representations, positive_representations, negative_representations)
#                 total_loss = total_loss+contrast_loss
#                 # print(y_pred)
#                 # print("total_loss",total_loss)
#                 if bool_gpm:
#                     total_loss.backward()
#                     # for i in range(len(feature_list)):
#                     #     Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
#                     #     feature_mat.append(Uf)
#                     bn_counter = 0
#                     for k, (m,params) in enumerate(model.named_parameters()):
#                         # print(params.grad)
#                         # print(m)
#                         if 'bn' not in m:
#                             k -= bn_counter
#                             sz =  params.grad.data.size(0)
#                             params.grad.data = torch.mul((params.grad.data - torch.mul(torch.mm(params.grad.data.view(sz,-1),\
#                                                     feature_mat[k]).view(params.size()),1)), (1))  
#                         else:
#                             bn_counter += 1    

#             else:       
#                 total_loss.backward()

            
            
#             opt.step() 
#             # teacher_model.load_state_dict(model.state_dict(), strict=False)
#             # gradient_rejection(model=model)
#             # if gradient_rejection.early_stop:
#             #     torch.save(model.state_dict(), check_point_file_name_norm)
#             train_losses.append(total_loss.item())

#             y_pred = y_pred.detach().cpu().numpy()
#             lab_y = lab_y.detach().cpu().numpy()
            
#             # lr_precision, lr_recall, _ = precision_recall_curve(lab_y, y_pred,pos_label=1)
#             # lr_auc_outlier =  auc(lr_recall, lr_precision)
            
        

#             # lr_precision, lr_recall, _ = precision_recall_curve(lab_y, [1-x for x in y_pred],pos_label=0)
#             # lr_auc_inliers =  auc(lr_recall, lr_precision)   
#             # prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f} - PR-AUC(inliers): {:.2f} - PR_auc(outlier)_curve {:.3f}'.format(
#             #      total_loss.item(), sup_loss.item(), distil_loss.item(), lr_auc_inliers,lr_auc_outlier ))
#             # r_auc = roc_auc_score(lab_y, y_pred)
#             # prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f}'.format(
#             #      total_loss, sup_loss, distil_loss))
#             prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f}'.format(
#                  total_loss.item(), sup_loss.item(), distil_loss.item()))
        
#         model.eval() # prep model for evaluation
#         val_pred,val_gt = [],[]
#         for data, target in valid_loader:
#             # pred = torch.argmax(model(data.to(device)),dim=1).reshape(target.shape)
#             pred = model(data.to(device))[:,1].reshape(target.shape)
#             y_pred = pred.detach().cpu().numpy().tolist()
#             val_pred.extend(y_pred)
#             val_gt.extend(target.detach().cpu().numpy().tolist())
#         lr_precision, lr_recall, _ = precision_recall_curve(val_gt, [x for x in val_pred], pos_label=1.)
#         lr_auc_minority =  auc(lr_recall, lr_precision)
#         # lr_precision, lr_recall, _ = precision_recall_curve(val_gt, val_pred, pos_label=1.)
#         # lr_auc_majority=  auc(lr_recall, lr_precision)
#         lr_auc = lr_auc_minority#[lr_auc_minority,lr_auc_majority]
#         # lr_auc = f1_score(val_gt,val_pred)
#             # calculate the loss
#             # loss = loss_fn(pred, target.to(device))
#             # record validation loss
#             # valid_losses.append(loss.item())
#         # valid_losses.append(np.nan_to_num(lr_auc))
#         # print training/validation statistics 
#         # calculate average loss over an epoch
#         train_loss = np.average(train_losses)
#         # valid_loss = np.average(valid_losses)
#         avg_train_losses.append(train_loss)
#         # avg_valid_losses.append(valid_loss)
#         epoch_len = len(str(epochs))
        
#         print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
#                      f'train_loss: {train_loss:.5f} ' +
#                      f'PR-AUC (I): {lr_auc:.5f}')
        
#         print(print_msg)
        
#         # clear lists to track next epoch
#         train_losses = []
#         valid_losses = []
        
#         # early_stopping needs the validation loss to check if it has decresed, 
#         # and if it has, it will make a checkpoint of the current model
#         early_stopping(lr_auc, model)
#         if early_stopping.counter <1:
#             scheduler.step()

#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
#     # load the last checkpoint with the best model
#     model.load_state_dict(torch.load(check_point_file_name))
#     teacher_model.load_state_dict(torch.load(check_point_file_name))

#     # temp_x,temp_y,temp_yname = X[labeled_indicies,:],y[labeled_indicies],y_classname[labeled_indicies]
    
#     # temp_x,temp_y,temp_yname = X[unlabeled_indicies,:],y[unlabeled_indicies],y_classname[unlabeled_indicies]
    

#     # temp_x,temp_y,temp_yname = labeled_X,labeled_y,labeled_y_classname
#     # if task_id > 0 and bool_reorganize_memory:
#     #     mem_start_time = time.time()
#     #     if str(mem_strat) == "replace":
            
#     #         tasks[0] = temp_x,temp_y,temp_yname
#     #         lab_samples_in_memory = split_a_task(tasks,lab_samp_in_mem_ratio)
#     #         tasks[0] = temp_x[lab_samples_in_memory[0],:],temp_y[lab_samples_in_memory[0]],temp_yname[lab_samples_in_memory[0]]
#     #         initialize_buffermemory(tasks=tasks,mem_size=memory_size)
#     #     elif str(mem_strat) == "equal":
            
#     #         memory_X, memory_y, memory_y_name = memory_update_equal_allocation2(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.15,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)
#     #     else:
            
#     #         memory_X, memory_y, memory_y_name = memory_update_equal_allocation(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.85,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)

#     #     mem_finish_time = time.time()
#     #     memory_population_time += mem_finish_time-mem_start_time

#     # mat_list = []    
#     temp_x,temp_y,temp_yname = X[labeled_indicies,:],y[labeled_indicies],y_classname[labeled_indicies]
#     # mat_list = get_representation_matrix (model, device, temp_x, temp_y)
#     if bool_gpm:
#         mat_list = get_representation_matrix (model, device, temp_x, temp_y,rand_samples=no_of_rand_samples)
#         feature_list = update_GPM(model, mat_list, threshold, feature_list) 
      
#     else:
#         feature_list = []

#     # grad_norm_dict[task_id] = grad_norm_list   
#     # print(grad_norm_dict)
#     if os.path.exists(check_point_file_name):
#         os.remove(check_point_file_name)
#     if os.path.exists(check_point_file_name_norm):
#         os.remove(check_point_file_name_norm) 

#     # print(f'Buffer memory size for task {task_id}: {memory_X.shape}')
    
#     return feature_list

     
TASK_CALIBRATED_THRESHOLD = {}   # task_id -> classification threshold chosen from that
                                  # task's OWN validation set (Youden's J), used by
                                  # testing() at report time instead of a blanket 0.5.
TASK0_ANCHOR_X = None
TASK0_ANCHOR_SOFT_TARGETS = None
TASK0_ANCHOR_WEIGHT = 0.0   # DIAGNOSTIC: anchor forward pass now fully skipped when this is 0

def train(str_train_model,tasks,task_class_ids,task_id,feature_list,threshold,X_val,y_val,bool_reorganize_memory,owl_data_labeling=False):
    
    global memory_X, memory_y, memory_y_name,local_count,global_count,local_store,input_shape,memory_size,task_num
    global classes_so_far,full,global_priority_list,local_priority_list,memory_population_time,replay_size
    global memory_population_time,epochs,grad_norm_dict,temp_norm
    global student_optimizer1,student_optimizer2,student_supervised_optimizer
    global teacher_model1,teacher_model2,teacher_supervised,student_model1,student_model2,student_supervised
    global truth_agreement_fraction_0, truth_agreement_fraction_1 
    global avg_CI, CI_list 
    global val_x_all_tasks, val_y_all_tasks
    global TASK0_ANCHOR_X, TASK0_ANCHOR_SOFT_TARGETS

    if str_train_model == "student1":
        model = student_model1
        opt = student_optimizer1
        teacher_model = teacher_model1
    elif str_train_model == "student2":
        model = student_model2
        opt = student_optimizer2
        teacher_model = teacher_model2
    elif str_train_model == "student_supervised":
        model = student_supervised
        opt = student_supervised_optimizer    
        teacher_model = teacher_supervised    

    grad_norm_list = []

    valid_loader = torch.utils.data.DataLoader(dataset(X_val,y_val),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=0)
    feature_mat = []
    X,y,y_classname = tasks[0][0],tasks[0][1],tasks[0][2]
    y_large,y_small = max(np.sum(y == 0),np.sum(y == 1)),min(np.sum(y == 0),np.sum(y == 1))
    print("majority class",y_large)
    print("minority class",y_small)
    print("class imbalance ratio",y_small/(y_large+y_small))
    # print("computed class imbalce ratio", clustering_class_imbalance(X))
    unique_y_classname = np.unique(y_classname)
    if unique_y_classname[0]%2 == 0:
        attack_y_name = unique_y_classname[0]
        benign_y_name = unique_y_classname[1]
    else:
        attack_y_name = unique_y_classname[1]
        benign_y_name = unique_y_classname[0]

    # if task_id > 0:
    #     compute_otdd(task_id, X, memory_X, memory_y_name, attack_y_name, benign_y_name)

    task_size = X.shape[0]
    # if True or owl_data_labeling == False: #for SPIDER
    if owl_data_labeling == False:

        if task_id == 0:
            labeled_indicies,unlabeled_indicies=split_a_task(tasks,0.99,task_class_ids)
        else:
            labeled_indicies,unlabeled_indicies=split_a_task(tasks,labels_ratio,task_class_ids)
        labeled_X,labeled_y,labeled_y_classname = X[labeled_indicies],y[labeled_indicies],y_classname[labeled_indicies]
        X_unlab,y_unlab,y_unlabclassname = X[unlabeled_indicies],y[unlabeled_indicies],y_classname[unlabeled_indicies]
        
        # Computing class imbalance in the labeled samples
        maj_class_count,min_class_count = max(np.sum(labeled_y == 0),np.sum(labeled_y == 1)),min(np.sum(labeled_y == 0),np.sum(labeled_y == 1))
        CI_list.append(min_class_count/(maj_class_count+min_class_count))
        # CI_list.append(maj_class_count/(min_class_count))
        # CI_list.append(np.sum(labeled_y == 0)/(np.sum(labeled_y == 1) + np.sum(labeled_y == 0)))
        # print("majority samples",maj_class_count)
        # print("minority samples",min_class_count)
        # CI_list.append(np.sum(labeled_y == 0)/(np.sum(labeled_y == 1) + np.sum(labeled_y == 0)))
        print(f'Class Imbalance for task {task_id} (% of class 0 samples)= {CI_list[-1]}\n')
        avg_CI = np.mean(CI_list)

        # Computing class imbalance ratio for the task (0:1)
        if task_id > 0:
            task_truth_agreement_fractions = owl_data_labeling_strategy(X, y, y_classname, unseen_task=False)
            print(f'\nCurrent task truth agreement fractions = {task_truth_agreement_fractions}')
            print(truth_agreement_fraction_0, truth_agreement_fraction_1)
            ## accumulation
            if truth_agreement_fraction_0 is None:
                truth_agreement_fraction_0 = task_truth_agreement_fractions[0] if task_truth_agreement_fractions[0] != 0 else None
            else:
                truth_agreement_fraction_0 = beta*truth_agreement_fraction_0 + (1 - beta)*task_truth_agreement_fractions[0] if task_truth_agreement_fractions[0] != 0 else truth_agreement_fraction_0
            
            if truth_agreement_fraction_1 is None:
                truth_agreement_fraction_1 = task_truth_agreement_fractions[1] if task_truth_agreement_fractions[1] != 0 else None
            else:
                truth_agreement_fraction_1 = beta*truth_agreement_fraction_1 + (1 - beta)*task_truth_agreement_fractions[1] if task_truth_agreement_fractions[1] != 0 else truth_agreement_fraction_1

            # truth_agreement_fraction_0, truth_agreement_fraction_1 = min(0.5,truth_agreement_fraction_0),min(0.5,truth_agreement_fraction_1)
            print(truth_agreement_fraction_0, truth_agreement_fraction_1)
        #     print()


    else:
        labeled_X, labeled_y, labeled_y_classname, X_unlab, labeled_indicies,unlabeled_indicies = owl_data_labeling_strategy(X, y, y_classname, unseen_task=True)
        
        # labeled_X_class_0,labeled_X_class_1,X_unlab = open_world_data_labeling(X)
        # labeled_X = np.concatenate((labeled_X_class_0,labeled_X_class_1),axis=0)
        # labeled_y = np.concatenate((np.zeros((labeled_X_class_0.shape[0],), dtype=np.int64),np.ones((labeled_X_class_1.shape[0],), dtype=np.int64)),axis=0)
        # labeled_y_classname = np.concatenate((np.full((labeled_X_class_0.shape[0],),fill_value=benign_y_name,dtype=np.int64),np.full((labeled_X_class_1.shape[0],),fill_value=attack_y_name,dtype=np.int64)),axis=0)
        # random_indices = np.random.permutation(len(labeled_X))
        # labeled_X,labeled_y,labeled_y_classname = X[random_indices],y[random_indices],y_classname[random_indices]
        # labeled_indicies = [lab_idx_num for lab_idx_num in range(labeled_X.shape[0])]
        # unlabeled_indicies = [unlab_idx_num+len(labeled_indicies) for unlab_idx_num in range(X_unlab.shape[0])]
        # labeled_indicies,unlabeled_indicies=open_world_data_labeling(X)
        # labeled_X,labeled_y,labeled_y_classname = X[labeled_indicies],y[labeled_indicies],y_classname[labeled_indicies]
        # X_unlab,y_unlab,y_unlabclassname = X[unlabeled_indicies],y[unlabeled_indicies],y_classname[unlabeled_indicies]
        
        # labeled_X_class_0,labeled_X_class_1,X_unlab = open_world_data_labeling(X)
        # print("expected class zero labels",labeled_X_class_0.shape)
        # print("actual class zero labels,",Counter(y[labeled_X_class_0]))
        # print("top 20 class 0 labels",y[labeled_X_class_0[0:20]])
        # print("expected class one labels",labeled_X_class_1.shape)
        # print("actual class one labels,",Counter(y[labeled_X_class_1]))
    
    
    if task_id > 0:
              
            mem_batch_size = floor(batch_size*b_m)
            rem_batch_size = batch_size-mem_batch_size
            # task_size = X.shape[0] + memory_X.shape[0] 
            labeled_batch_size = floor(rem_batch_size*labels_ratio)
            unlabeled_batch_size = rem_batch_size - (labeled_batch_size)
            no_of_labeled_batches = floor(len(labeled_indicies)/labeled_batch_size)
            #no_of_batches = floor(len(labeled_indicies)/labeled_batch_size)
            no_of_unlab_batches = floor(len(unlabeled_indicies)/unlabeled_batch_size)
            p = np.random.permutation(labeled_X.shape[0])
            labeled_X,labeled_y,labeled_y_classname = labeled_X[p,:],labeled_y[p],labeled_y_classname[p]
            no_of_batches = max(no_of_labeled_batches,no_of_unlab_batches)
            print(f"mem_batch:{mem_batch_size}_{labeled_batch_size}_{unlabeled_batch_size},labeled batchs:{no_of_batches} and unlabaled batches:{no_of_unlab_batches}")
            # exit()
            
    else:
        # initialize_buffermemory(labeled_task,memory_size)
        task_size = X.shape[0]    
        mem_batch_size = floor(batch_size*b_m)
        rem_batch_size = batch_size-mem_batch_size
        labeled_batch_size = rem_batch_size
        no_of_batches = floor(task_size/rem_batch_size)
        unlabeled_batch_size=2
        no_of_labeled_batches = no_of_batches
        # labeled_batch_size = floor(batch_size*0.99)
        # labeled_batch_size = floor(batch_size*labels_ratio)
        # unlabeled_batch_size = batch_size-labeled_batch_size
        # no_of_batches = floor(task_size/batch_size)

    if bool_gpm:
        for i in range(len(feature_list)):
            Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
            feature_mat.append(Uf)    
    

    ###Buffer memory organization
    temp_x,temp_y,temp_yname = labeled_X,labeled_y,labeled_y_classname
    if task_id > 0 and bool_reorganize_memory:
        mem_start_time = time.time()
        if str(mem_strat) == "replace":
            
            tasks[0] = temp_x,temp_y,temp_yname
            lab_samples_in_memory = split_a_task(tasks,lab_samp_in_mem_ratio)
            tasks[0] = temp_x[lab_samples_in_memory[0],:],temp_y[lab_samples_in_memory[0]],temp_yname[lab_samples_in_memory[0]]
            initialize_buffermemory(tasks=tasks,mem_size=memory_size)
        elif str(mem_strat) == "equal":
            
            memory_X, memory_y, memory_y_name = memory_update_equal_allocation2(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.15,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)
        else:
            
            memory_X, memory_y, memory_y_name = memory_update_equal_allocation(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.85,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)

        mem_finish_time = time.time()
        memory_population_time += mem_finish_time-mem_start_time


    ###Training encoder for self-supervision
    # print("************Training the Encoder***********")
    # encoder = vime_self(device,X_unlab, p_m=0.3, alpha=2.0, parameters={'epochs': 5, 'batch_size': 32})
    # encoder = encoder.eval()


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
                # if task_id > 0 and batch_idx < no_of_unlab_batches:
                    # unlabeled_X = torch.from_numpy(X_unlab[batch_idx*unlabeled_batch_size:batch_idx*unlabeled_batch_size+unlabeled_batch_size]).to(device)
                    unlabeled_X = torch.from_numpy(X_unlab[batch_idx*unlabeled_batch_size:batch_idx*unlabeled_batch_size+unlabeled_batch_size]).to(device)
                else:
                    rand_indices = list(random.sample(range(X_unlab.shape[0]),min(unlabeled_batch_size,X_unlab.shape[0])))
                    unlabeled_X = torch.from_numpy(X_unlab[rand_indices]).to(device)
                    # rand_indices = list(random.sample(range(X_unlab.shape[0]),min(unlabeled_batch_size,X_unlab.shape[0])))
                    # unlabeled_X = torch.from_numpy(X_unlab[rand_indices]).to(device)


                # if image_resolution is not None:
                #     unlabeled_X = unlabeled_X.reshape(image_resolution)
                unlabeled_pred = torch.softmax(model(unlabeled_X),dim=1).detach()
                if task_id >= 0 and batch_idx < no_of_labeled_batches:
                    lab_X = labeled_X[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]  
                    lab_y = labeled_y[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]
                else:
                    rand_indices = list(random.sample(range(labeled_X.shape[0]),min(labeled_batch_size,labeled_X.shape[0])))    
                    lab_X = labeled_X[rand_indices]  
                    lab_y = labeled_y[rand_indices]
            
            
            # lab_X = labeled_X[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]  
            # lab_y = labeled_y[batch_idx*labeled_batch_size:batch_idx*labeled_batch_size+labeled_batch_size]
            task_lab_X,task_lab_y = lab_X,lab_y
            if task_id >=0:
                
                mem_batch = sample_batch_from_memory(floor(batch_size*b_m),minority_alloc=batch_minority_alloc)
                if mem_batch is not None and mem_batch[0].shape[0] > 0:
                    
                    lab_X = np.concatenate((lab_X,mem_batch[0]), axis=0)  
                    # temp_mem_X = torch.from_numpy(mem_batch[0]).to(device)
                    # if image_resolution is not None:
                    #     temp_mem_X = temp_mem_X.reshape(image_resolution)
                    # temp_mem_y = torch.argmax(teacher_model(temp_mem_X),dim=1).detach().cpu().numpy().squeeze()
                    
                    # lab_y = np.concatenate((lab_y,temp_mem_y), axis=0)
                    
                    lab_y = np.concatenate((lab_y,mem_batch[1]), axis=0)
            lab_X = torch.from_numpy(lab_X).to(device)
            # if image_resolution is not None:
            #         lab_X = lab_X.reshape(image_resolution)
                   
            # print(model(lab_X))
            y_pred = torch.softmax(model(lab_X),dim=1).squeeze()#.to(device)                      
            lab_y = torch.from_numpy(lab_y).to(device).to(dtype=torch.long)#.reshape(y_pred.shape)

            # lab_y = F.one_hot(lab_y, 2)
            sup_loss = loss_fn(y_pred.float(),F.one_hot(lab_y.to(dtype=torch.long), 2).float())#to(device)
            # sup_loss = loss_fn(y_pred,lab_y.float())
            total_loss = sup_loss
            
            distil_loss = 0
            distil_loss = torch.as_tensor(distil_loss).to(device)
            opt.zero_grad()
            if task_id > 0:
                if str_train_model!="student_supervised":
                    #computing the distillation loss
                    # distil_loss_list = compute_distill_loss(unlabeled_pred,unlabeled_X)
                    distil_loss_list = compute_distill_loss_with_confidence(unlabeled_pred,unlabeled_X)
                    distil_loss = distil_loss_list[0]

                    # distil_loss = compute_distill_loss_self_supervision(p_m=0.3, K=3,unlabeled_x=unlabeled_X,encoder_model=encoder)
                    total_loss = total_loss +  alpha *distil_loss  
                    # lab_X = torch.cat((lab_X,unlabeled_X),0)
                    # lab_y = torch.cat((lab_y,distil_loss_list[1]),0)

                contrast_loss = 0
                
                if bool_closs:
                    # positives, negatives = construct_positive_negative_samples(lab_X, lab_y)  
                    positives, negatives = construct_positive_negative_samples_from_memory(task_lab_y)  
                    anchor_representations = model(torch.from_numpy(task_lab_X).to(device))
                    positive_representations = model(positives)
                    negative_representations = model(negatives)
                    contrast_loss = contrastive_loss(anchor_representations, positive_representations, negative_representations)
                total_loss = total_loss+contrast_loss

                # grad.data = grad - grad_proj
                if bool_gpm:
                    total_loss.backward()

                    # TASK0 OUTPUT ANCHOR loss (LwF-style): penalize the current
                    # model's predictions on a random mini-batch of task 0's own
                    # validation data for drifting away from what task 0's
                    # just-trained (frozen) model predicted on the same samples.
                    # Only active from task 1 onward. Computed and backpropagated
                    # SEPARATELY from the main task loss above, with its OWN
                    # gradient-norm clip, before being combined with the main
                    # gradient and GPM-projected together. This targets ONLY the
                    # anchor term's contribution, leaving the main task's gradient
                    # untouched -- confirmed necessary: an earlier version that
                    # clipped the COMBINED gradient (main+anchor together) fixed
                    # cicids2018's collapse but regressed cicids2017 (seen AUT
                    # 0.98->0.93), because it rescaled the main task's legitimate
                    # gradient along with anchor's every time the combined norm
                    # exceeded the threshold. Clipping anchor's own contribution
                    # specifically is the targeted fix for the mechanism actually
                    # identified as the cause (see memory notes for the isolation
                    # evidence), without diluting anything else.
                    # DIAGNOSTIC TEST: skip the anchor forward pass ENTIRELY
                    # when its weight is 0, instead of just zeroing its loss
                    # contribution -- isolates whether the extra forward pass
                    # itself (which updates BatchNorm running stats regardless
                    # of loss weight, since BN updates on any training-mode
                    # forward call) is the actual mechanism, independent of
                    # anchor loss's gradient/value.
                    if TASK0_ANCHOR_X is not None and TASK0_ANCHOR_WEIGHT != 0:
                        main_grads = [p.grad.clone() if p.grad is not None else None
                                      for p in model.parameters()]
                        opt.zero_grad()
                        t0_anchor_idx = np.random.choice(TASK0_ANCHOR_X.shape[0],
                                                          size=min(batch_size, TASK0_ANCHOR_X.shape[0]),
                                                          replace=False)
                        t0_anchor_pred = torch.softmax(model(TASK0_ANCHOR_X[t0_anchor_idx].to(device)), dim=1)
                        t0_anchor_target = TASK0_ANCHOR_SOFT_TARGETS[t0_anchor_idx]
                        t0_anchor_loss = F.mse_loss(t0_anchor_pred, t0_anchor_target)
                        (TASK0_ANCHOR_WEIGHT * t0_anchor_loss).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        for p, mg in zip(model.parameters(), main_grads):
                            if p.grad is not None and mg is not None:
                                p.grad = p.grad + mg
                            elif mg is not None:
                                p.grad = mg

                    bn_counter = 0
                    for k, (m,params) in enumerate(model.named_parameters()):
                        if 'bn' not in m:
                            k -= bn_counter
                            sz =  params.grad.data.size(0)
                            params.grad.data = torch.mul((params.grad.data - torch.mul(torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[k]).view(params.size()),1)), (1))
                        else:
                            bn_counter += 1

            else:
                # bool_gpm=False path -- never exercised in this whole
                # investigation (every run uses --bool_gpm=True), kept as the
                # simpler pre-existing behavior (anchor loss folded into
                # total_loss, no separate clip) rather than restructured
                # without the ability to validate it.
                if TASK0_ANCHOR_X is not None:
                    t0_anchor_idx = np.random.choice(TASK0_ANCHOR_X.shape[0],
                                                      size=min(batch_size, TASK0_ANCHOR_X.shape[0]),
                                                      replace=False)
                    t0_anchor_pred = torch.softmax(model(TASK0_ANCHOR_X[t0_anchor_idx].to(device)), dim=1)
                    t0_anchor_target = TASK0_ANCHOR_SOFT_TARGETS[t0_anchor_idx]
                    t0_anchor_loss = F.mse_loss(t0_anchor_pred, t0_anchor_target)
                    total_loss = total_loss + TASK0_ANCHOR_WEIGHT * t0_anchor_loss
                total_loss.backward()

            opt.step()
            # teacher_model.load_state_dict(model.state_dict(), strict=False)
            # gradient_rejection(model=model)
            # if gradient_rejection.early_stop:
            #     torch.save(model.state_dict(), check_point_file_name_norm)
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
            # prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f}'.format(
            #      total_loss, sup_loss, distil_loss))
            prog_bar.set_description('loss: {:.5f} - sup: {:.5f} - dist_loss: {:.5f}'.format(
                 total_loss.item(), sup_loss.item(), distil_loss.item()))
        
        model.eval() # prep model for evaluation
        # MAXIMIN checkpoint selection (Formula 1): score each task 0..task_id
        # (INCLUDING current) on its OWN validation slice (PR-AUC), use the
        # MINIMUM per-task PR-AUC as the EarlyStopping criterion, instead of
        # one PR-AUC pooled across all tasks trained so far (which can hide a
        # regression on an early task behind a larger, newer task's data).
        # PR-AUC (not a composite with FPR/FNR at a fixed threshold) is used
        # deliberately: it is threshold-invariant and degrades gracefully
        # under extreme class imbalance, whereas any 0.5-threshold-based
        # count metric (FPR/FNR) has variance that blows up as the minority
        # class count shrinks -- confirmed via direct diagnosis: cicids2018
        # has a task with ~91 total minority-class samples, and a composite
        # score including FPR/FNR there produced catastrophically noisy
        # checkpoint decisions via the min() over tasks. FNR itself is
        # handled separately and explicitly by the end-of-training threshold
        # calibration (see taskwise_lazytrain()), so nothing is lost by
        # keeping this checkpoint-selection criterion to PR-AUC alone.
        per_task_scores = []
        per_task_val_aucs = []
        with torch.no_grad():
            for vt in range(task_id + 1):
                vt_X, vt_y = val_x_all_tasks[vt], val_y_all_tasks[vt]
                vt_loader = torch.utils.data.DataLoader(dataset(vt_X, vt_y),
                                                         batch_size=batch_size, num_workers=0)
                vt_pred, vt_gt = [], []
                for data, target in vt_loader:
                    pred = model(data.to(device))[:, 1].reshape(target.shape)
                    vt_pred.extend(pred.detach().cpu().numpy().tolist())
                    vt_gt.extend(target.detach().cpu().numpy().tolist())
                vt_precision, vt_recall, _ = precision_recall_curve(vt_gt, vt_pred, pos_label=1.)
                vt_prauc = auc(vt_recall, vt_precision)
                per_task_val_aucs.append(vt_prauc)
                per_task_scores.append(vt_prauc)
        lr_auc = min(per_task_scores)
        lr_auc_minority = lr_auc
        print(f"[MAXIMIN-CKPT] task={task_id} per_task_val_PRAUC={['%.4f'%v for v in per_task_val_aucs]} "
              f"per_task_composite={['%.4f'%v for v in per_task_scores]} -> min={lr_auc:.4f}")
        # lr_precision, lr_recall, _ = precision_recall_curve(val_gt, val_pred, pos_label=1.)
        # lr_auc_majority=  auc(lr_recall, lr_precision)
        lr_auc = lr_auc_minority#[lr_auc_minority,lr_auc_majority]
        # lr_auc = f1_score(val_gt,val_pred)
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
    teacher_model.load_state_dict(torch.load(check_point_file_name))

    # TASK0 OUTPUT ANCHOR snapshot: right after task 0's own training, freeze
    # its predictions on its own validation set as a soft-target anchor for
    # every later task's training loss. Validation set only, never test set.
    if task_id == 0:
        model.eval()
        with torch.no_grad():
            anchor_X = val_x_all_tasks[0]
            anchor_loader = torch.utils.data.DataLoader(dataset(anchor_X, np.zeros(anchor_X.shape[0])),
                                                         batch_size=batch_size, num_workers=0)
            soft_targets = []
            for data, _ in anchor_loader:
                out = torch.softmax(model(data.to(device)), dim=1)
                soft_targets.append(out.detach())
            TASK0_ANCHOR_X = torch.from_numpy(anchor_X).float()
            TASK0_ANCHOR_SOFT_TARGETS = torch.cat(soft_targets, dim=0)
        print(f"[TASK0-ANCHOR] snapshotted {TASK0_ANCHOR_X.shape[0]} task-0 validation samples "
              f"as a soft-target anchor for later tasks' training")

    # temp_x,temp_y,temp_yname = X[labeled_indicies,:],y[labeled_indicies],y_classname[labeled_indicies]
    
    # temp_x,temp_y,temp_yname = X[unlabeled_indicies,:],y[unlabeled_indicies],y_classname[unlabeled_indicies]
    

    # temp_x,temp_y,temp_yname = labeled_X,labeled_y,labeled_y_classname
    # if task_id > 0 and bool_reorganize_memory:
    #     mem_start_time = time.time()
    #     if str(mem_strat) == "replace":
            
    #         tasks[0] = temp_x,temp_y,temp_yname
    #         lab_samples_in_memory = split_a_task(tasks,lab_samp_in_mem_ratio)
    #         tasks[0] = temp_x[lab_samples_in_memory[0],:],temp_y[lab_samples_in_memory[0]],temp_yname[lab_samples_in_memory[0]]
    #         initialize_buffermemory(tasks=tasks,mem_size=memory_size)
    #     elif str(mem_strat) == "equal":
            
    #         memory_X, memory_y, memory_y_name = memory_update_equal_allocation2(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.15,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)
    #     else:
            
    #         memory_X, memory_y, memory_y_name = memory_update_equal_allocation(temp_x,temp_y,temp_yname,memory_size,memory_X, memory_y, memory_y_name,minorityclass_ids,majority_class_memory_share=0.85,random_sample_selection=True,temp_model=model,image_resolution=image_resolution,device=device)

    #     mem_finish_time = time.time()
    #     memory_population_time += mem_finish_time-mem_start_time

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

    # print(f'Buffer memory size for task {task_id}: {memory_X.shape}')
    
    return feature_list

     






     
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
    global test_x,test_y,task_num,task_order,auc_result,val_x_all_tasks,val_y_all_tasks
    global teacher_model1,teacher_model2,teacher_supervised,student_model1,student_model2,student_supervised
    global owl_self_labelled_count_class_0, owl_self_labelled_count_class_1
    global owl_analyst_labelled_count_class_0, owl_analyst_labelled_count_class_1
    global avg_CI, CI_list
    global truth_agreement_fraction_0, truth_agreement_fraction_1
    global training_cutoff
    global TASK_CALIBRATED_THRESHOLD
    global SELF_LABEL_ACCURACY_LOG
    # global consecutive_otdd

    # random.shuffle(task_order)
    print("task order",task_order)
    threshold = np.array([0.95,0.99,0.99,0.98,0.99,0.99,0.99,0.98,0.99,0.98])
    feature_list_student1,feature_list_student2,feature_list_student_supervised =[],[],[]

    train_order = task_order[:training_cutoff]
    test_order = task_order[training_cutoff:]

    # for task_id,task in enumerate(train_order):
    #     task_class_ids = []
    #     task_minorityclass_ids = []
    #     for class_ in task:
    #         task_class_ids.extend([class_])
    #         if class_ in minorityclass_ids:
    #             task_minorityclass_ids.extend([class_])
    #     # print("loading task:",task_id)     
    #     input_shape,tasks,X_test,y_test,X_val,y_val = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=bool_encode_benign,bool_encode_anomaly=bool_encode_anomaly,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=load_whole_train_data)
    #     val_x_all_tasks.extend([X_val])
    #     val_y_all_tasks.extend([y_val])

    print(f'\nTraining on first {training_cutoff} tasks...')
    for task_id,task in enumerate(train_order):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("\nloading task:",task_id)     
        input_shape,tasks,X_test,y_test,X_val,y_val = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=bool_encode_benign,bool_encode_anomaly=bool_encode_anomaly,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=load_whole_train_data)
        # print(tasks)

        if ds == 'anoshift':
            train_len,val_len,test_len = (tasks[0][0]).shape[0],(X_val).shape[0],(X_val).shape[0]
            # Combine the files into a single array
            merged_data = np.concatenate((tasks[0][0], X_val, X_test), axis=0)

            # Perform min-max normalization
            min_val = np.min(merged_data, axis=0)
            max_val = np.max(merged_data, axis=0)
            normalized_data = (merged_data - min_val) / (max_val - min_val)

            # Split the normalized data back into three files with original sizes
            temp_tasks = [(normalized_data[:train_len],tasks[0][1],tasks[0][2])]
            tasks = temp_tasks
            # tasks[0][0] = (normalized_data[:train_len])
            X_val = normalized_data[train_len:train_len + val_len]
            X_test = normalized_data[train_len + val_len:]
        

        val_x_all_tasks.extend([X_val])
        val_y_all_tasks.extend([y_val])
        # val_x_all_tasks,val_y_all_tasks = [X_val],[y_val]
        print("validation dataset size",X_val.shape)

        test_x.extend([X_test])
        test_y.extend([y_test])
        print("Training task:",task_id)
        task_num = task_id
        if task_num == int(0):
            initialize_buffermemory(tasks=tasks,mem_size=memory_size)


        if mlps == 1:
            feature_list_student1 =train("student1",tasks,task_class_ids,task_id,feature_list_student1,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),True,False)
                    
        elif mlps == 2:
            feature_list_student1 =train("student1",tasks,task_class_ids,task_id,feature_list_student1,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),True,False)
            feature_list_student2 =train("student2",tasks,task_class_ids,task_id,feature_list_student2,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),False,False)
            
        
        else:
            feature_list_student1 =train("student1",tasks,task_class_ids,task_id,feature_list_student1,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),True,False)
            feature_list_student2 =train("student2",tasks,task_class_ids,task_id,feature_list_student2,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),False,False)
            feature_list_student_supervised =train("student_supervised",tasks,task_class_ids,task_id,feature_list_student_supervised,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),False,False)
        # teacher_model.load_state_dict(model.state_dict())
    # testing(training_cutoff=training_cutoff, seen_data=True) 
    # testing(training_cutoff=training_cutoff, seen_data=False)    
        
    print(f'\nOpen world setting training from task {training_cutoff} onwards...')
    # BUGFIX (ported, see cicids2017_spider_owl_neurips2024_2.py): preserve None
    # rather than clamping it away, so owl_data_labeling_strategy()'s own
    # pre-existing "no evidence yet -> full trust" fallback can execute.
    truth_agreement_fraction_0 = min(truth_agreement_fraction_0, 0.5) if truth_agreement_fraction_0 is not None else None
    truth_agreement_fraction_1 = min(truth_agreement_fraction_1, 0.1) if truth_agreement_fraction_1 is not None else None
    print(f'Agreement fraction: class 0 = {truth_agreement_fraction_0}, class 1 = {truth_agreement_fraction_1}')
    
    avg_CI = np.mean(CI_list)
    print(f'Average CI over training tasks (% of class 0 samples)= {avg_CI}')

    for task_id,task in enumerate(test_order,start=training_cutoff):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("\nloading task:",task_id)     
        input_shape,tasks,X_test,y_test,X_val,y_val = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=bool_encode_benign,bool_encode_anomaly=bool_encode_anomaly,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=load_whole_train_data)
        
        if ds == 'anoshift':
            train_len,val_len,test_len = (tasks[0][0]).shape[0],(X_val).shape[0],(X_val).shape[0]
            # Combine the files into a single array
            merged_data = np.concatenate((tasks[0][0], X_val, X_test), axis=0)

            # Perform min-max normalization
            min_val = np.min(merged_data, axis=0)
            max_val = np.max(merged_data, axis=0)
            normalized_data = (merged_data - min_val) / (max_val - min_val)

            # Split the normalized data back into three files with original sizes
            temp_tasks = [(normalized_data[:train_len],tasks[0][1],tasks[0][2])]
            tasks = temp_tasks
            # tasks[0][0] = (normalized_data[:train_len])
            X_val = normalized_data[train_len:train_len + val_len]
            X_test = normalized_data[train_len + val_len:]

        val_x_all_tasks.extend([X_val])
        val_y_all_tasks.extend([y_val])
        # val_x_all_tasks.extend([X_val])
        # val_y_all_tasks.extend([y_val])
        # val_x_all_tasks,val_y_all_tasks = [X_val],[y_val]
        print("validation dataset size",X_val.shape)

        test_x.extend([X_test])
        test_y.extend([y_test])
        print("Training task:",task_id)
        task_num = task_id
        if task_num == int(0):
            initialize_buffermemory(tasks=tasks,mem_size=memory_size)

        if mlps == 1:
            # continue
            feature_list_student1 =train("student1",tasks,task_class_ids,task_id,feature_list_student1,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),True,True)
                    
        elif mlps == 2:
            feature_list_student1 =train("student1",tasks,task_class_ids,task_id,feature_list_student1,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),True,True)
            feature_list_student2 =train("student2",tasks,task_class_ids,task_id,feature_list_student2,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),False,True)
            
        
        else:
            feature_list_student1 =train("student1",tasks,task_class_ids,task_id,feature_list_student1,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),True,True)
            feature_list_student2 =train("student2",tasks,task_class_ids,task_id,feature_list_student2,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),False,True)
           #feature_list_student_supervised =train("student_supervised",tasks,task_class_ids,task_id,feature_list_student_supervised,threshold,np.concatenate(val_x_all_tasks, axis=0 ),np.concatenate(val_y_all_tasks, axis=0 ),False,False)
        # teacher_model.load_state_dict(model.state_dict())
        # teacher_model.load_state_dict(model.state_dict())
    # testing(training_cutoff=training_cutoff, seen_data=True) 
    # testing(training_cutoff=training_cutoff, seen_data=False)  
    
    training_cutoff=5
    # THRESHOLD CALIBRATION (v5): once, at the very end of all training, from
    # the truly final model state. Per task 1..N-1 (task 0 excluded -- see
    # cicids2017_spider_owl_neurips2024_2.py history for why), pick the
    # threshold maximizing Youden's J = TPR-FPR on that task's own validation
    # set, shrunk 60% toward 0.5. Validation set only, never test set.
    student_model1.eval()
    with torch.no_grad():
        for vt in range(len(task_order)):
            if vt == 0:
                continue
            vt_X, vt_y = val_x_all_tasks[vt], val_y_all_tasks[vt]
            vt_loader = torch.utils.data.DataLoader(dataset(vt_X, vt_y),
                                                     batch_size=batch_size, num_workers=0)
            vt_pred, vt_gt = [], []
            for data, target in vt_loader:
                pred = student_model1(data.to(device))[:, 1].reshape(target.shape)
                vt_pred.extend(pred.detach().cpu().numpy().tolist())
                vt_gt.extend(target.detach().cpu().numpy().tolist())
            fpr_arr, tpr_arr, roc_thresh = roc_curve(vt_gt, vt_pred, pos_label=1)
            j_scores = tpr_arr - fpr_arr
            best_idx = int(np.argmax(j_scores))
            raw_thresh = float(np.clip(roc_thresh[best_idx], 0.01, 0.99))
            THRESH_SHRINKAGE = 0.6
            TASK_CALIBRATED_THRESHOLD[vt] = 0.5 + THRESH_SHRINKAGE * (raw_thresh - 0.5)
    print(f"[THRESH-CALIB] final thresholds (all tasks, end of training): "
          f"{ {k: round(v,4) for k,v in TASK_CALIBRATED_THRESHOLD.items()} }")

    test_set_results = []
    with open(temp_filename, 'w') as fp:
        test_set_results.extend([testing(training_cutoff=training_cutoff, seen_data=True),testing(training_cutoff=training_cutoff, seen_data=False),str(owl_self_labelled_count_class_0), str(owl_self_labelled_count_class_1),str(owl_analyst_labelled_count_class_0), str(owl_analyst_labelled_count_class_1),testing(training_cutoff=len(task_order), seen_data=True), SELF_LABEL_ACCURACY_LOG ])
#         test_set_results.extend([ ##enable when tsne is required
#     testing_tsne(training_cutoff=training_cutoff, seen_data=True),
#     testing_tsne(training_cutoff=training_cutoff, seen_data=False),
#     str(owl_self_labelled_count_class_0),
#     str(owl_self_labelled_count_class_1),
#     str(owl_analyst_labelled_count_class_0),
#     str(owl_analyst_labelled_count_class_1),
#     testing_tsne(training_cutoff=len(task_order), seen_data=True)
# ])
        auc_result[str(args.seed)] = test_set_results
        json.dump(auc_result, fp)

    # print('\nOTDD values for consecutive tasks:')
    # for i, val in enumerate(consecutive_otdd):
    #     print(f'Task ({i},{i + 1}): {val}')
    # print()

    print('************** OWL labelling stats ****************')
    print(f'Total number of self-labelled samples = {owl_self_labelled_count_class_0 + owl_self_labelled_count_class_1}')
    print(f'Class 0 count = {owl_self_labelled_count_class_0}')
    print(f'Class 1 count = {owl_self_labelled_count_class_1}')

    print(f'Total number of analyst-labelled samples = {owl_analyst_labelled_count_class_0 + owl_analyst_labelled_count_class_1}')
    print(f'Class 0 count = {owl_analyst_labelled_count_class_0}')
    print(f'Class 1 count = {owl_analyst_labelled_count_class_1}')

    # ── SELF-LABEL ACCURACY: per-task, per-class, and single-seed aggregate ──
    print('\n************** SELF-LABEL ACCURACY (this seed) ****************')
    sl_rows = []
    macro_accs, micro_correct, micro_total = [], 0, 0
    for rec in SELF_LABEL_ACCURACY_LOG:
        c0c, c0t = rec['class0_correct'], rec['class0_total']
        c1c, c1t = rec['class1_correct'], rec['class1_total']
        c0_acc = c0c / c0t if c0t > 0 else float('nan')
        c1_acc = c1c / c1t if c1t > 0 else float('nan')
        task_total = c0t + c1t
        task_acc = (c0c + c1c) / task_total if task_total > 0 else float('nan')
        sl_rows.append([rec['task_id'], f'{c0_acc*100:.2f}% ({c0c}/{c0t})',
                         f'{c1_acc*100:.2f}% ({c1c}/{c1t})', f'{task_acc*100:.2f}% ({c0c+c1c}/{task_total})'])
        if task_total > 0:
            macro_accs.append(task_acc)
        micro_correct += (c0c + c1c)
        micro_total += task_total
    print(tabulate(sl_rows, headers=['task', 'class0 (benign) acc', 'class1 (attack) acc', 'combined acc'], tablefmt='grid'))
    macro_avg = float(np.mean(macro_accs)) if macro_accs else float('nan')
    micro_avg = micro_correct / micro_total if micro_total > 0 else float('nan')
    print(f'Single-seed self-label accuracy summary: macro (mean over tasks) = {macro_avg*100:.2f}%, '
          f'micro (sample-weighted over all self-labeled samples) = {micro_avg*100:.2f}%')


        
def testing(training_cutoff, seen_data=False):

    dataset_loadtime=0
    global TASK_CALIBRATED_THRESHOLD
    global teacher_model1,teacher_model2,teacher_supervised
    global student_model1,student_model2,student_supervised

    

    
    if mlps == 1:
        models = [student_model1]
    elif mlps == 2:
        models = [student_model1,student_model2]
    else:
        models = [student_model1,student_model2,student_supervised]
    
    # ensemble_main.eval()
    
    task_CI_pnt = []
    test_CI_pnt =[]
    prauc_in_pnt = []
    prauc_out_pnt = []
    en_prauc_in_pnt = []
    en_prauc_out_pnt = []
    fpr_pnt = []   # FPR per task (FP / (FP + TN))
    fnr_pnt = []   # FNR per task (FN / (FN + TP))

    if seen_data:
        testing_tasks = task_order[:training_cutoff]
        start_id = 0
        
    else:
        testing_tasks  = task_order[training_cutoff:]
        start_id = training_cutoff
        

    # for task_id,task in enumerate(task_order):#[training_cutoff:], start = training_cutoff):
    # for task_id,task in enumerate(task_order[training_cutoff:], start = training_cutoff):
    for task_id,task in enumerate(testing_tasks, start = start_id):
        
        
        task_class_ids = []
        task_minorityclass_ids = []
        
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        start = time.time()
        # input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=0,bool_encode_anomaly=1,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=True)
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=bool_encode_benign,bool_encode_anomaly=bool_encode_anomaly,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=False)
        
        if seen_data:
            features,target_label = X_test,y_test
        else:
            features,target_label = X_test,y_test
            # features,target_label = tasks[0][0],tasks[0][1]
        print(f'testing on task {task_id}: {features.shape}')
        
        dataset_loadtime += time.time()-start
        
        
        # valid_loader = torch.utils.data.DataLoader(dataset(tasks[0][0],tasks[0][1]),
        #                                        batch_size=batch_size,
        #                                     #    sampler=valid_sampler,
        #                                        num_workers=0)     
        valid_loader = torch.utils.data.DataLoader(dataset(features,target_label),
                                               batch_size=batch_size,
                                            #    sampler=valid_sampler,
                                               num_workers=0)
        
        val_pred, en_val_pred, val_actual = [],[], []
        for data, target in valid_loader:
            class_probs = [] 
            with torch.no_grad():
                for model in models:
                    outputs = torch.softmax(model(data.to(device)), dim=1)
                    class_probs.append(outputs)

            pred = torch.stack(class_probs).mean(dim=0)[:,1].reshape(target.shape)
            
            # pred = (model(data.to(device))[:,1]).reshape(target.shape)
            en_pred = pred
            # en_pred = ensemble_main(data.to(device)).reshape(target.shape)
            y_pred = pred.detach().cpu().numpy().tolist()
            en_y_pred = en_pred.detach().cpu().numpy().tolist()

            val_pred.extend(y_pred)
            en_val_pred.extend(en_y_pred)
            val_actual.extend(target.detach().cpu().numpy().tolist())         

        
        train_y = tasks[0][1]

        # print(f'test set size:= {len(val_actual)} with {len([i for i in val_actual if int(i) == 1])} attacks')
        # print(f'train set size:= {len(tasks[0][1])} with {len([i for i in tasks[0][1] if int(i) ==1])} attacks')
        
        # train_CI = len([i for i in tasks[0][1] if round(i) == 1])/(len(tasks[0][1])-len([i for i in tasks[0][1] if round(i) == 1]))
        # test_CI = len([i for i in val_actual if round(i) == 1])/(len(val_actual)-len([i for i in val_actual if round(i) == 1]))
        # print(f'test CI: {test_CI}')
        # print(f'train CI: {train_CI}')
        
        precision, recall, thresholds = precision_recall_curve(val_actual, val_pred,pos_label=1.0)
        auc_precision_recall_1 = auc(recall, precision)
        precision, recall, thresholds = precision_recall_curve(val_actual, en_val_pred)
        en_auc_precision_recall_1 = auc(recall, precision)

        precision, recall, thresholds = precision_recall_curve(val_actual, [1-val for val in val_pred], pos_label=0.)
        auc_precision_recall_0 = auc(recall, precision)
        precision, recall, thresholds = precision_recall_curve(val_actual, [1-val for val in en_val_pred], pos_label=0.)
        en_auc_precision_recall_0 = auc(recall, precision)

        #when number of 1s > 0s then the 1 is the inliers and0 is the outliers
        # auc_precision_recall_in = auc_precision_recall_0 if test_CI < 1 else auc_precision_recall_1
        # auc_precision_recall_out = auc_precision_recall_1 if test_CI < 1 else auc_precision_recall_0

        # en_auc_precision_recall_in = en_auc_precision_recall_0 if test_CI < 1 else en_auc_precision_recall_1
        # en_auc_precision_recall_out = en_auc_precision_recall_1 if test_CI < 1 else en_auc_precision_recall_0

        # task_CI_pnt.append(train_CI)
        # test_CI_pnt.append(test_CI)
        # prauc_in_pnt.append(auc_precision_recall_in)
        # prauc_out_pnt.append(auc_precision_recall_out)
        # en_prauc_in_pnt.append(en_auc_precision_recall_in)
        # en_prauc_out_pnt.append(en_auc_precision_recall_out)
        prauc_in_pnt.append(auc_precision_recall_0)
        prauc_out_pnt.append(auc_precision_recall_1)
        en_prauc_in_pnt.append(en_auc_precision_recall_0)
        en_prauc_out_pnt.append(en_auc_precision_recall_1)

        # Compute FPR and FNR at threshold 0.5
        task_thresh = TASK_CALIBRATED_THRESHOLD.get(task_id, 0.5)
        val_pred_binary = [1 if p >= task_thresh else 0 for p in val_pred]
        tn, fp, fn, tp = confusion_matrix(val_actual, val_pred_binary, labels=[0, 1]).ravel()
        fpr_pnt.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        fnr_pnt.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)

        # print(f'prauc inliers: {auc_precision_recall_in}')
        # print(f'prauc outliers: {auc_precision_recall_out}')
        # print('')

    N = len(testing_tasks) #number of test tasks
    prauc_in_aut  = 0
    prauc_out_aut = 0

    if N<2:
        print('not printing AUT values since it requires atleast 2 test tasks')
        # indices: [0]=prauc_in_pnt, [1]=prauc_out_pnt, [2]=prauc_in_aut, [3]=prauc_out_aut,
        #          [4]=training_cutoff, [5]=seen_data, [6]=N, [7]=fpr_pnt, [8]=fnr_pnt
        return [prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N,fpr_pnt,fnr_pnt]
    
    
    for i in range(N-1):
        prauc_in_aut+= (prauc_in_pnt[i]+prauc_in_pnt[i+1])/(2)
        prauc_out_aut+=(prauc_out_pnt[i]+prauc_out_pnt[i+1])/(2)
    prauc_in_aut  = prauc_in_aut/(N-1)
    prauc_out_aut = prauc_out_aut/(N-1)
    
    print(f'AUT(prauc inliers,{N}) := {prauc_in_aut}')
    print(f'AUT(prauc outliers,{N}) := {prauc_out_aut}')

    print('\npnt table for SPIDER:')
    pnt_table = [
        # ['task_CI']+ task_CI_pnt, 
        # ['test_CI'] + test_CI_pnt,
        ['prauc Benign traffic'] + prauc_in_pnt, 
        ['prauc Attack traffic'] + prauc_out_pnt
    ]
    print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
    
    # print('\npnt table for Ensemble models:')
    # pnt_table = [
    #     # ['task_CI']+ task_CI_pnt, 
    #     # ['test_CI'] + test_CI_pnt,
    #     ['prauc Benign traffic'] + en_prauc_in_pnt, 
    #     ['prauc Attack traffic'] + en_prauc_out_pnt
    # ]
    # print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
    print(f'dataset loading time: {dataset_loadtime}s\n')
    
    # print('Here json dump of the data for easy unparsing')
    # print(f'#pnt_table#{json.dumps(pnt_table)}#end_pnt_table#')
    # print(f'#train_order#{json.dumps(train_order)}#end_train_order#')
    # indices: [0]=prauc_in_pnt, [1]=prauc_out_pnt, [2]=prauc_in_aut, [3]=prauc_out_aut,
    #          [4]=training_cutoff, [5]=seen_data, [6]=N, [7]=fpr_pnt, [8]=fnr_pnt
    return [prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N,fpr_pnt,fnr_pnt]

def testing_tsne(training_cutoff, seen_data=False):

    import os
    import time
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from sklearn.manifold import TSNE
    from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
    from tabulate import tabulate

    dataset_loadtime = 0

    global student_model1, student_model2, student_supervised

    # =========================
    # Model selection
    # =========================
    if mlps == 1:
        models = [student_model1]
    elif mlps == 2:
        models = [student_model1, student_model2]
    else:
        models = [student_model1, student_model2, student_supervised]

    # =========================
    # Metric containers
    # =========================
    prauc_in_pnt, prauc_out_pnt = [], []
    en_prauc_in_pnt, en_prauc_out_pnt = [], []

    # === NEW: FPR / FNR containers ===
    fpr_per_task = []
    fnr_per_task = []
    task_ids = []

    # =========================
    # Task selection
    # =========================
    if seen_data:
        testing_tasks = task_order[:training_cutoff]
        start_id = 0
    else:
        testing_tasks = task_order[training_cutoff:]
        start_id = training_cutoff

    tsne_root = f"./tsne_results/{ds}/GPM_{bool_gpm}/cutoff_{training_cutoff}"
    os.makedirs(tsne_root, exist_ok=True)

    # =========================
    # Task-wise evaluation
    # =========================
    for task_id, task in enumerate(testing_tasks, start=start_id):

        task_class_ids, task_minorityclass_ids = [], []
        for class_ in task:
            task_class_ids.append(class_)
            if class_ in minorityclass_ids:
                task_minorityclass_ids.append(class_)

        start = time.time()
        input_shape, tasks, X_test, y_test, _, _ = load_dataset(
            pth,
            task_class_ids,
            task_minorityclass_ids,
            tasks_list,
            task2_list,
            [task],
            bool_encode_benign=bool_encode_benign,
            bool_encode_anomaly=bool_encode_anomaly,
            label=label,
            bool_create_tasks_avalanche=False,
            load_whole_train_data=False
        )
        dataset_loadtime += time.time() - start

        features, target_label = X_test, y_test

        valid_loader = torch.utils.data.DataLoader(
            dataset(features, target_label),
            batch_size=batch_size,
            num_workers=0
        )

        val_pred, val_actual = [], []

        # === Embeddings for t-SNE ===
        task_embeddings = []
        task_labels = []

        for data, target in valid_loader:

            class_probs = []

            with torch.no_grad():
                for model in models:
                    out = torch.softmax(model(data.to(device)), dim=1)
                    class_probs.append(out)

                # last hidden layer embeddings
                emb = models[0].act['hidden6']
                task_embeddings.append(emb.detach().cpu())
                task_labels.append(target.detach().cpu())

            pred = torch.stack(class_probs).mean(dim=0)[:, 1].reshape(target.shape)

            val_pred.extend(pred.cpu().numpy().tolist())
            val_actual.extend(target.cpu().numpy().tolist())

        # =========================
        # PR-AUC
        # =========================
        precision, recall, _ = precision_recall_curve(val_actual, val_pred, pos_label=1.0)
        auc_attack = auc(recall, precision)

        precision, recall, _ = precision_recall_curve(
            val_actual, [1 - v for v in val_pred], pos_label=0.0
        )
        auc_benign = auc(recall, precision)

        prauc_in_pnt.append(auc_benign)
        prauc_out_pnt.append(auc_attack)
        en_prauc_in_pnt.append(auc_benign)
        en_prauc_out_pnt.append(auc_attack)

        # =========================
        # FPR / FNR computation
        # =========================
        binary_pred = (np.array(val_pred) >= 0.5).astype(int)
        binary_true = np.array(val_actual).astype(int)

        tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred).ravel()

        fpr = fp / (fp + tn + 1e-8)
        fnr = fn / (fn + tp + 1e-8)

        fpr_per_task.append(fpr)
        fnr_per_task.append(fnr)
        task_ids.append(task_id)

        # =========================
        # t-SNE visualization
        # =========================
        task_embeddings = torch.cat(task_embeddings, dim=0).numpy()
        task_labels = torch.cat(task_labels, dim=0).numpy()

        max_points = 5000
        if task_embeddings.shape[0] > max_points:
            idx = np.random.choice(task_embeddings.shape[0], max_points, replace=False)
            task_embeddings = task_embeddings[idx]
            task_labels = task_labels[idx]

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
            init="pca",
            random_state=42
        )

        emb_2d = tsne.fit_transform(task_embeddings)

        plt.figure(figsize=(6, 5))
        plt.scatter(
            emb_2d[task_labels == 0, 0],
            emb_2d[task_labels == 0, 1],
            s=8,
            alpha=0.35,
            c="#4C72B0",
            label="Benign"
        )
        plt.scatter(
            emb_2d[task_labels == 1, 0],
            emb_2d[task_labels == 1, 1],
            s=10,
            alpha=0.85,
            c="#DD8452",
            edgecolors="black",
            linewidths=0.2,
            label="Attack"
        )
        plt.legend(frameon=False)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"t-SNE projections of test set (Task {task_id})")
        plt.tight_layout()
        plt.savefig(f"{tsne_root}/tsne_task_{task_id}.pdf", bbox_inches="tight")
        plt.close()

   # =========================
# FPR / FNR BAR PLOT (ANNOTATED, POLISHED)
# =========================
    fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(task_ids)), 6))

    bar_w = 0.45
    x_pos = np.arange(len(task_ids))

    # Color-blind safe palette (Okabe–Ito)
    # fpr_color = "#0072B2"   # deep blue
    # fnr_color = "#D55E00"   # vermillion
    fpr_color = "#009E73"   # green
    fnr_color = "#D55E00"   # vermillion

    # Plot bars
    fpr_bars = ax.bar(
        x_pos - bar_w / 2,
        fpr_per_task,
        width=bar_w,
        color=fpr_color,
        edgecolor='black',
        hatch='//',
        linewidth=1.0,
        label="False Positive Rate (FPR)"
    )

    fnr_bars = ax.bar(
        x_pos + bar_w / 2,
        fnr_per_task,
        width=bar_w,
        color=fnr_color,
        edgecolor="black",
        linewidth=1.0,
        hatch='xx',
        label="False Negative Rate (FNR)"
    )

    # ---- Annotate values on bars (SAFE placement) ----
    def annotate_bars(bars):
        for p in bars:
            height = p.get_height()
            # cap annotation to stay inside plot
            y_text = min(height + 0.04, 1.45)

            ax.annotate(
                f"{height:.2f}",
                xy=(p.get_x() + p.get_width() / 2, y_text),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold"
            )

    annotate_bars(fpr_bars)
    annotate_bars(fnr_bars)

    # Axes & formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(task_ids, rotation=0,fontweight='bold',fontsize=20)
    ax.set_xlabel("Task ID", fontsize=20,fontweight = 'bold')
    ax.set_ylabel("Rate", fontsize=20,fontweight = 'bold')

    # 🔒 Add headroom so labels never clip
    ax.set_ylim(0.0, 1.5)
    plt.setp(ax.get_yticklabels(), fontsize=20, fontweight='bold')


    # ax.set_title(
    #     "False Positive Rate (FPR) and False Negative Rate (FNR) Across Tasks",
    #     fontsize=13,
    #     pad=10
    # )

    # 🔒 Legend INSIDE plot, top-right
    ax.legend(
    frameon=False,
    fontsize=20,
    loc="upper right"
)


    # Subtle grid for readability
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(f"{tsne_root}/fpr_fnr_per_task.pdf", bbox_inches="tight")
    plt.close()



    # =========================
    # AUT computation
    # =========================
    N = len(testing_tasks)
    prauc_in_aut, prauc_out_aut = 0, 0

    if N > 1:
        for i in range(N - 1):
            prauc_in_aut += (prauc_in_pnt[i] + prauc_in_pnt[i + 1]) / 2
            prauc_out_aut += (prauc_out_pnt[i] + prauc_out_pnt[i + 1]) / 2

        prauc_in_aut /= (N - 1)
        prauc_out_aut /= (N - 1)

    print(f"AUT(prauc benign,{N}) := {prauc_in_aut}")
    print(f"AUT(prauc attack,{N}) := {prauc_out_aut}")

    print("\nPNT table:")
    pnt_table = [
        ['PR-AUC Benign'] + prauc_in_pnt,
        ['PR-AUC Attack'] + prauc_out_pnt
    ]

    print(tabulate(
        pnt_table,
        headers=[''] + [str(training_cutoff + i) if not seen_data else str(i) for i in range(N)],
        tablefmt='grid'
    ))

    print(f"dataset loading time: {dataset_loadtime:.2f}s\n")

    # Return structure matches testing() so main files can read both interchangeably:
    # [0]=prauc_in_pnt, [1]=prauc_out_pnt, [2]=prauc_in_aut, [3]=prauc_out_aut,
    # [4]=training_cutoff, [5]=seen_data, [6]=N, [7]=fpr_per_task, [8]=fnr_per_task
    return [
        prauc_in_pnt,
        prauc_out_pnt,
        prauc_in_aut,
        prauc_out_aut,
        training_cutoff,
        seen_data,
        N,
        fpr_per_task,
        fnr_per_task
    ]



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
    offset = 25000000
    # offset = 25000
    for idx in range(0,X_test.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(X_test[idx1:idx2,:].astype(float)).to(device)
        if image_resolution is not None:
                    X_test1 = X_test1.reshape(image_resolution)
        # temp = torch.argmax(model(X_test1.float()),dim=1).detach().cpu().numpy()
        temp = (model(X_test1.float())[:,1]).detach().cpu().numpy()
        if idx1==0:
            yhat = temp
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
    return compute_results(y_test,yhat)
    # print("test sample counters are",Counter(y_test))

import numpy as np

def select_random_indices_for_classes(labels, class_label_1, class_label_2, num_samples_per_class1=1000,num_samples_per_class2=100):
    indices_class_label_1 = np.where(labels == class_label_1)[0]
    indices_class_label_2 = np.where(labels == class_label_2)[0]
    print("classs 0",len(indices_class_label_1))
    print("classs 0",len(indices_class_label_2))

    random_indices_class_label_1 = np.random.choice(indices_class_label_1, num_samples_per_class1, replace=False).tolist()
    random_indices_class_label_2 = np.random.choice(indices_class_label_2, num_samples_per_class2, replace=False).tolist()
    random_indices_class_label_1.extend(random_indices_class_label_2)
    return random_indices_class_label_1




def tsne_visualize(seed,labels_ratio=0.1,batch_minority=0.5,rand_samples=100,ppt=50):
    global X_test,y_test
    test_embeddings = torch.zeros((0,10), dtype=torch.float32)
    if pth_testset is not None:
        X_test,y_test = load_teset(pth_testset,testset_class_ids,label)
    yhat = None    
    model.eval()
    indices = select_random_indices_for_classes(y_test,0,1,10000,100000)
    print(indices)
    X_test,y_test = X_test[indices],y_test[indices]
    print("computing the results")
    offset = 25000
    for idx in range(0,X_test.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(X_test[idx1:idx2,:].astype(float)).to(device)
        if image_resolution is not None:
                    X_test1 = X_test1.reshape(image_resolution)
        temp = model(X_test1.float()).detach().cpu().numpy()
        embeddings = model.act['hidden6']
        if idx1==0:
            yhat = temp
            
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
        test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)    
    test_embeddings = np.array(test_embeddings)
    dir_struct = {0:"tsne",1:"caring",2:str(label)}    
    dir_struct[3 ]= "_lab_ratio_"+str(labels_ratio)+"_minorty_"+str(batch_minority)+"_rand_samp_"+str(rand_samples)+"_seed"+str(seed)
    for pt in [5,10,25,50,100,150,200,350,500,750,1000,1500,2000,2500,5000,10000,20000,40000,50000]:
        plot_tsne(y_test,yhat,test_embeddings,dir_struct,pt)

def plot_grdient_norm_line_graph():
    dir_struct = {0:"line_graph",1:"caring",2:str(label)}    
    dir_struct[3 ]= "_lab_ratio_"+str(labels_ratio)+"_minorty_"+str(b_m)+"_bool_gpm"+str(bool_gpm)+"_rand_samp_"+str(no_of_rand_samples)+"_seed"+str(seed)
    plot_grad_norm_line_graph(dir_struct,grad_norm_dict)




def start_execution(dataset_name,l_rate,w_decay):
    global input_shape,tasks,X_test,y_test,test_x,test_y,val_x_all_tasks,val_y_all_tasks
    start_time=time.time()
    load_metadata(dataset_name,l_rate,w_decay)
    # load_model_metadata()
    # print(model)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("number of parameters is",pytorch_total_params)
    print(f'Is lazy training: {is_lazy_training}')
    if is_lazy_training:
        test_x,test_y = [],[]
        val_x_all_tasks,val_y_all_tasks = [],[]
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





def cleanup_stale_checkpoint_files():
    """Remove checkpoint*.pt files left behind by a crashed/killed previous
    run (creating PID no longer alive). These accumulate because a hard kill
    (timeout, ctrl-C, OOM-kill) skips the normal end-of-task
    os.remove(check_point_file_name) cleanup in train()/train_and_gradient().
    Only removes a file once its PID is confirmed dead -- never touches a
    live PID's checkpoint, so concurrent runs on the same machine are safe.
    """
    for fname in os.listdir('.'):
        if not (fname.startswith('checkpoint') and fname.endswith('.pt')):
            continue
        middle = fname[len('checkpoint'):-len('.pt')]
        pid_str = middle[:-len('grad_norm')] if middle.endswith('grad_norm') else middle
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        try:
            os.kill(pid, 0)
            # PID exists, but a zombie (defunct, never reaped) process is not
            # doing any work and will never clean up its own checkpoint --
            # treat it as dead too.
            with open(f'/proc/{pid}/status') as pf:
                if 'State:\tZ' not in pf.read():
                    continue  # genuinely alive -- leave it alone
        except (OSError, FileNotFoundError):
            pass
        try:
            os.remove(fname)
            print(f"[CHECKPOINT-CLEANUP] removed stale {fname} (pid {pid} not running)")
        except OSError:
            pass


if __name__ == "__main__":
    cleanup_stale_checkpoint_files()
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--ds', type=str, default="ids18", metavar='S',help='dataset name')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)')
    parser.add_argument('--filename', type=str,default="temp", metavar='S',help='json file name')
    parser.add_argument('--b_m', type=float, default=0.2, metavar='S',help='batch memory ratio(default: 0.2)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='S',help='batch memory ratio(default: 0.001)')
    parser.add_argument('--wd', type=float, default= 1e-04, metavar='S',help='batch memory ratio(default: 0.01)')
    parser.add_argument('--label_ratio', type=float, default=0.1, metavar='S',help='labeled ratio (default: 0.1)')
    parser.add_argument('--nps', type=int, metavar='S',default=10000,help='number of projection samples(default: 100)')
    parser.add_argument('--bma', type=float, metavar='S',default=0.3,help='batch minority allocation(default: 0.8)')
    parser.add_argument('--alpha', type=float, metavar='S',default=9,help='distill loss multiplier(default: 9)')
    parser.add_argument('--lab_samp_in_mem_ratio', type=float, metavar='S',default=0.1,help='Percentage of labeled samples to store in memory(default: 1.0)')
    parser.add_argument('--bool_gpm', type=str, metavar='S',default="True",help='Enables gradient projections(default: True)')
    parser.add_argument('--mem_strat', type=str, metavar='S',default="equal",help='Buffer memory strategy(default: full initialization)')
    parser.add_argument('--training_cutoff', type=int, default=5, metavar='S',help='train the model for first n tasks and test for time decay on the rest')
    parser.add_argument('--bool_closs', type=str, metavar='S',default="False",help='Enables using contrastive loss(default: False)')
    parser.add_argument('--mlps', type=int, metavar='S',default=1,help='Number of learners (MLPs)default: 1)')
    parser.add_argument('--train_with_unlab', type=str, metavar='S',default="True",help='Sets to train with unlabeled data(default: True)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N', help='number of training epochs/task (default: 10)')
    parser.add_argument('--beta', type=float, metavar='S',default=0.1,help='hyperparameter for accumulation of agreement fraction')
    parser.add_argument('--cos_dist', type=float, metavar='S',default=0.1,help='cosine distance for OWL(default: 0.1)')
    parser.add_argument('--mode_val', type=int, metavar='S',default=98,help='Mode value for OWL (default: 98)')


    args = parser.parse_args()
    set_seed(args.seed)
    get_gpu(args.gpu)
    print("seed is",args.seed)
    global labels_ratio,no_of_rand_samples,l_rate,w_decay,batch_minority_allocation,b_m,alpha,lab_samp_in_mem_ratio,bool_gpm,mem_strat,temp_filename,auc_result,seed,bool_closs,mlps,training_cutoff, epochs, ds, beta, train_with_unlab,cos_dist_ip, mode_value
    epochs = args.n_epochs
    b_m = float(args.b_m)
    labels_ratio=float(args.label_ratio)
    no_of_rand_samples = int(args.nps)
    batch_minority_alloc = float(args.bma)
    alpha = float(args.alpha)
    l_rate = float(args.lr)
    w_decay = float(args.wd)
    lab_samp_in_mem_ratio = float(args.lab_samp_in_mem_ratio)
    train_with_unlab = eval(args.train_with_unlab)
    bool_gpm = eval(args.bool_gpm)
    bool_closs = eval(args.bool_closs)
    mem_strat = str(args.mem_strat)
    mlps = int(args.mlps)
    mode_value = int(args.mode_val)
    cos_dist_ip = float(args.cos_dist)
    training_cutoff = int(args.training_cutoff)
    seed = args.seed
    ppt = 25
    ds = args.ds
    beta = args.beta
    print("{:<20}  {:<20}".format('Argument','Value'))
    print("*"*80)
    for arg in vars(args):
        print("{:<20}  {:<20}".format(arg, getattr(args, arg)))
    print("*"*80)    
    auc_result= {}
    temp_filename = str(args.filename)    
    start_execution(args.ds,l_rate,w_decay)
    # print("seed is",args.seed)
    # grad_norm_mean = sum(grad_norm_dict)/len(grad_norm_dict)
    # grad_norm_variance = statistics.variance(grad_norm_dict)
    # print("grad norm avg:",grad_norm_mean)
    # print("grad norm variance",grad_norm_variance)
    grad_norm_mean,grad_norm_variance = 0,0
    
    # with open(temp_filename, 'w') as fp:
    #     test_set_results = evaluate_on_testset()
    #     test_set_results.extend([grad_norm_mean,grad_norm_variance])
    #     auc_result[str(args.seed)] = test_set_results
    #     json.dump(auc_result, fp)
    
    print("*"*80)
    
    
    # evaluate_on_testset()
    # tsne_visualize(args.seed,labels_ratio,batch_minority_alloc,no_of_rand_samples,ppt)


    
