from turtle import st
# from neurips_2022.CBRS.anoshift_AGEM import X_train
import torch
import numpy as np
import warnings
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset
from torchmetrics import Accuracy
from torch.optim import Adam
import avalanche
from avalanche.benchmarks.generators import tensors_benchmark

from avalanche.benchmarks.generators import tensor_scenario
from avalanche.benchmarks.scenarios.classification_scenario import GenericCLScenario
from avalanche.evaluation.metrics import (
    ExperienceForgetting,
    StreamConfusionMatrix,
    accuracy_metrics,
    loss_metrics,
    
)
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin
from avalanche.training import EWC, AGEM, Naive, LwF, SynapticIntelligence, GSS_greedy,GEM
from avalanche.evaluation.metric_definitions import GenericPluginMetric, Metric



from utils.customdataloader import load_dataset,avalanche_tensor_to_tensor,get_inputshape,compute_total_minority_testsamples
from utils.metrics import compute_results
from utils.utils import log,create_directories,trigger_logging,set_seed,get_gpu,load_model, get_dataset_info
from utils.config.configurations import cfg
# from utils.config.configurations import cicids2017 as ds
from utils.metadata import initialize_metadata
from sklearn.metrics import precision_recall_curve,auc,confusion_matrix



import time
import random
from collections import Counter
import json
from tabulate import tabulate
from sys import getsizeof as size
from typing import (
    Sequence,
    Union,
    Any,
    Tuple,
    Dict,
    Optional,
    Iterable,
    NamedTuple,
)
from memory_profiler import memory_usage


## TODO: create a separate model for avalanche (with softmax as output)




memory_population_time=0

pattern_per_exp,task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,pth_testset = None,None,None,None,None,None,None,None,None
batch_size,device = None,None
test_x,test_y = [],[]

cl_strategy,model,opt,loss_fn,train_acc_metric,learning_rate,is_lazy_training = None,None,None,None,None,None,None
nc = 0
no_tasks = 0
bool_create_tasks_avalanche = True
warnings.filterwarnings("ignore")

class dataset(Dataset):

    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
 
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.length

def load_metadata(dataset_name,lr,w_decay): 
    log('loading meta data')   
    global task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,no_tasks,pattern_per_exp,cl_strategy
    global replay_size,memory_size,minority_allocation,epochs,batch_size,device,learning_rate,is_lazy_training,pth_testset
    global cl_strat, ds

    print(avalanche.__version__)
    # set_seed(125)
    # get_gpu()
    ds = get_dataset_info(dataset_name)
    label = ds.label
    cfg.avalanche_dir = True    
    no_tasks = ds.no_tasks
    metadata_dict = initialize_metadata(label)
    temp_dict = metadata_dict[no_tasks]
    task_order = temp_dict['task_order']
    class_ids = temp_dict['class_ids']
    minorityclass_ids = temp_dict['minorityclass_ids']
    pth = temp_dict['path']  
    tasks_list = temp_dict['tasks_list']
    task2_list = temp_dict['task2_list']
    replay_size = ds.replay_size
    memory_size = ds.mem_size
    minority_allocation = ds.minority_allocation
    epochs = ds.n_epochs
    batch_size = ds.batch_size
    device = cfg.device
    learning_rate = lr#ds.learning_rate
    no_tasks = ds.no_tasks
    pattern_per_exp = ds.pattern_per_exp
    is_lazy_training = ds.is_lazy_training
    compute_total_minority_testsamples(pth=pth,dataset_label=label,minorityclass_ids=minorityclass_ids,no_tasks=no_tasks)
    load_model_metadata(w_decay)
    strategy = set_cl_strategy_name(cl_strat)
    print(f'CL strategy = {strategy}')
    create_directories(label)
    trigger_logging(label=label)
    cl_strategy = get_cl_strategy(strategy)
    # if strategy == 2:
    #     task_order = temp_dict['task_order2']

    
    

def load_model_metadata(w_decay):
    log("loading model parameter")
    global model,opt,loss_fn,train_acc_metric
    
    model = load_model(label=label,inputsize=get_inputshape(pth,class_ids), softmax=True)
    print(model)
    model = model.to(device)
    # model.train()
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=w_decay)
    # opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    # loss_fn = torch.nn.BCELoss()
    # Labels here are one-hot (2-class) floats (create_tasks_avalanche's
    # F.one_hot(...).float()), but plain CrossEntropyLoss expects integer
    # class-index targets in this PyTorch version for the code path PyTorch
    # picks with these shapes -- crashes with "nll_loss_forward... not
    # implemented for 'Float'". This never surfaced before because nothing
    # ever ran evaluation in the original files (no early stopping/periodic
    # eval existed); eval is the first thing to actually hit this. Wrapping
    # to convert one-hot floats to class indices makes both paths correct.
    def _one_hot_ce_loss(pred, target):
        if target.dim() > 1:
            target = target.argmax(dim=1)
        return torch.nn.functional.cross_entropy(pred, target.long())
    loss_fn = _one_hot_ce_loss
    train_acc_metric = Accuracy().to(device)




def create_avalanche_scenario(tasks_dataset,task_labels):
    
    generic_scenario = tensors_benchmark(train_tensors = tasks_dataset[0],
                                           test_tensors = tasks_dataset[1],#tasks_dataset[1],
                                           task_labels = task_labels,
                                        )

    return generic_scenario  

 

def train_a_lazytask(train_scenario,task_label,task_id):
   
    global ds

    generic_scenario = create_avalanche_scenario([train_scenario[0],train_scenario[1]],task_labels=task_label)
    for task_number, experience in enumerate(generic_scenario.train_stream):
            # eval_streams is what makes Avalanche actually run periodic
            # (per-epoch) evaluation during training -- without it, nothing
            # triggers PRAUCStreamMetric or EarlyStoppingPlugin, and
            # train_epochs would just run to its ceiling unconditionally,
            # same as before this fix. generic_scenario.test_stream is this
            # task's own held-out validation split (see the comment on
            # create_tasks_avalanche's return values -- what's positionally
            # called "test" in this scenario is genuinely validation data,
            # confirmed separately from the true held-out test set that
            # testing() loads independently for final reporting).
            res = cl_strategy.train(experience, eval_streams=[generic_scenario.test_stream])

    if ds.enable_checkpoint:
        checkpoint_location = str(cfg.outputdir) + '/models' +'/task_'+str(task_id)+ '.th'
        # print("location:",checkpoint_location)
        torch.save({'epoch': 5, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()}, checkpoint_location)                  


# ── VALIDATION-BASED EARLY STOPPING for the Avalanche baselines ────────────
# The original file trains every strategy (EWC/AGEM/GEM/GSS-greedy/SI) for a
# fixed ds.n_epochs (5 for cicids2018), with no validation-based stopping --
# unlike the SPIDER/OWL pipeline, which always early-stops on validation
# attack-class PR-AUC (patience=3). Monitoring PR-AUC here too, not the
# built-in accuracy/loss metrics, for a specific fairness reason: these
# baselines exist to be fairly compared against SPIDER/OWL, and giving one
# side a stopping rule suited to this severely imbalanced attack-detection
# setting while the other gets a generic one would confound the comparison
# (is a baseline losing to the method, or to a worse-suited stopping rule?).
# Every other part of the protocol (data, architecture, batch size) is
# already held constant between the two -- this keeps the stopping rule
# constant too.
class _PRAUCAccumulator(Metric):
    """Accumulates (prediction, target) pairs across an eval stream and
    computes attack-class (label=1) PR-AUC over the whole accumulation --
    PR-AUC is not incrementally averageable like accuracy, so it has to be
    computed once, in full, from every collected point."""

    def __init__(self):
        self._preds = []
        self._targets = []

    def reset(self) -> None:
        self._preds = []
        self._targets = []

    def update(self, preds, targets) -> None:
        self._preds.extend(preds)
        self._targets.extend(targets)

    def result(self):
        if len(set(self._targets)) < 2:
            # A degenerate eval batch (only one class present) can't define
            # a PR curve -- return the neutral/worst score rather than
            # crashing, so early stopping just sees "no improvement" here.
            return 0.0
        precision, recall, _ = precision_recall_curve(self._targets, self._preds, pos_label=1.0)
        return float(auc(recall, precision))


class PRAUCStreamMetric(GenericPluginMetric):
    """Reports attack-class PR-AUC once per full eval stream pass (matching
    the same reset_at/emit_at='stream' semantics EarlyStoppingPlugin expects
    for its default accuracy metric) -- emits under the name 'PRAUC_Stream'."""

    def __init__(self):
        self._accumulator = _PRAUCAccumulator()
        super().__init__(self._accumulator, reset_at="stream", emit_at="stream", mode="eval")

    def update(self, strategy):
        with torch.no_grad():
            preds = torch.softmax(strategy.mb_output, dim=1)[:, 1].detach().cpu().numpy().tolist()
        targets = strategy.mb_y.detach().cpu().numpy()
        # mb_y is one-hot (2-class) here, same encoding create_tasks_avalanche
        # uses everywhere else in this file -- take the attack-class column.
        if targets.ndim > 1:
            targets = targets[:, 1]
        targets = targets.tolist()
        self._accumulator.update(preds, targets)

    def __str__(self):
        return "PRAUC"


def evaluation_plugin():
    # log to text file
    text_logger = TextLogger(open(f"{cfg.outputdir}/logs/log.txt", "w+"))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
                                    # accuracy_metrics/loss_metrics/ExperienceForgetting
                                    # removed too: same underlying issue as
                                    # StreamConfusionMatrix below -- Avalanche's own
                                    # built-in metrics assume integer class-index
                                    # targets, but labels here are one-hot floats
                                    # (create_tasks_avalanche's F.one_hot(...).float()).
                                    # Training-mode use of these apparently never hit
                                    # the bad path, but their eval-mode computation
                                    # does (CUDA assert: target index out of range) --
                                    # again, this only surfaces now because eval never
                                    # ran in the original file. PRAUCStreamMetric below
                                    # handles the one-hot encoding correctly and is
                                    # what actually drives early stopping; the removed
                                    # metrics were diagnostic-only, not required.
                                    # StreamConfusionMatrix removed: it has its own
                                    # pre-existing device-mismatch bug (internal
                                    # _cm_tensor stays on CPU, indexed with GPU
                                    # tensors) that only surfaces once eval actually
                                    # runs, which never happened in the original file.
                                    # Not needed for early stopping (PRAUCStreamMetric
                                    # below is what drives that); dropped rather than
                                    # patching Avalanche's own library code.
                                    PRAUCStreamMetric(),
                                    loggers=[interactive_logger, text_logger],
                                   )

    return eval_plugin


def set_cl_strategy_name(strategy_id):
    if strategy_id == 1:
        cfg.clstrategy = "GEM"            
    elif strategy_id == 0:
        cfg.clstrategy = "EWC"
    elif strategy_id == 2:
        cfg.clstrategy = "AGEM" 
    elif strategy_id == 3:
        cfg.clstrategy = "GSS-greedy"   
    elif strategy_id == 4:
        cfg.clstrategy = "LwF"    
    return strategy_id    



# Fixed ds.n_epochs (5, from utils/config/configurations.py, left untouched
# per instruction) is no longer the actual stopping point -- early stopping
# decides that now. This is just a generous ceiling to bound worst-case
# runtime; patience=3 (same value the SPIDER/OWL pipeline uses) is what
# actually determines when training stops in the normal case.
EARLY_STOP_MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 3


def _early_stopping_plugin():
    return EarlyStoppingPlugin(patience=EARLY_STOP_PATIENCE, val_stream_name="test_stream",
                                metric_name="PRAUC", mode="max")
    # NOTE: metric_name must match PRAUCStreamMetric.__str__() ("PRAUC"), not
    # a "_Stream" suffix -- see ctu13_avalanche_2.py for the full trace of
    # why (EarlyStoppingPlugin prefix-matches metric_name + "/eval_phase/" +
    # val_stream_name against emitted names; "_Stream" belongs to the
    # built-in metrics' own __str__, not something appended by the plugin).


def get_cl_strategy(strategy_id):
    strategy = None
    global input_shape

    if cfg.clstrategy == "GEM":
        strategy = GEM(model,
                  optimizer = opt,
                  patterns_per_exp = pattern_per_exp,
                  criterion = loss_fn,
                  train_mb_size = batch_size,
                  train_epochs = EARLY_STOP_MAX_EPOCHS,
                  eval_mb_size = 128,
                  evaluator = evaluation_plugin(),
                  plugins = [_early_stopping_plugin()],
                  eval_every = 1,
                  device = device,
                  )

    elif cfg.clstrategy == "EWC":
        strategy = EWC(
           model,
           optimizer=opt,
           ewc_lambda=0.001,
           criterion=loss_fn,
           train_mb_size=batch_size,
           train_epochs=EARLY_STOP_MAX_EPOCHS,
           eval_mb_size=128,
           evaluator=evaluation_plugin(),
           plugins=[_early_stopping_plugin()],
           eval_every=1,
           device=device,
       )

    elif cfg.clstrategy =="AGEM" :
        strategy = AGEM(model,
                  optimizer = opt,
                  patterns_per_exp = pattern_per_exp,
                  criterion = loss_fn,
                  train_mb_size = batch_size,
                  train_epochs = EARLY_STOP_MAX_EPOCHS,
                  eval_mb_size = 128,
                  evaluator = evaluation_plugin(),
                  plugins = [_early_stopping_plugin()],
                  eval_every = 1,
                  device = device,
                  )

    elif cfg.clstrategy == "GSS-greedy":
        print("shape is:",get_inputshape(pth,class_ids))
        strategy = GSS_greedy( model,
                   optimizer=Adam(model.parameters()),
                   mem_size=replay_size,
                  criterion=loss_fn,
                  train_mb_size=batch_size,
                  train_epochs=EARLY_STOP_MAX_EPOCHS,
                  eval_mb_size=128,
                  input_size=[get_inputshape(pth,class_ids)],
                  evaluator=evaluation_plugin(),
                  plugins=[_early_stopping_plugin()],
           eval_every=1,
                 device=device,
                )

    elif cfg.clstrategy == "LwF":
        strategy = SynapticIntelligence(
           model,
           optimizer=opt,
           si_lambda=0.1,
        #    alpha=1, temperature=2,
           criterion=loss_fn,
           train_mb_size=batch_size,
           train_epochs=EARLY_STOP_MAX_EPOCHS,
           eval_mb_size=128,
           evaluator=evaluation_plugin(),
           plugins=[_early_stopping_plugin()],
           eval_every=1,
           device=device,
       )

    print(strategy)
                                                   

    return strategy          

def train(train_scenario,cl_strategy):
    for task_number, experience in enumerate(train_scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        # print(type(experience))
        # eval_streams triggers periodic (per-epoch) evaluation -- see the
        # matching comment in train_a_lazytask for why this is needed for
        # early stopping to actually fire.
        res = cl_strategy.train(experience, eval_streams=[train_scenario.test_stream])
        if ds.enable_checkpoint:
            checkpoint_location = str(cfg.outputdir) + '/models' +'/task_'+str(task_number)+ '.th'
            # print("location:",checkpoint_location)
            torch.save({'epoch': 5, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()}, checkpoint_location) 


def get_balanced_testset(X,y):
    X_test,y_test = X,y
    bool_idx_list = list()
    no_of_ones= np.count_nonzero(y_test == 1)
    c=0
    for idx in range(X_test.shape[0]):
        if y_test[idx] == 0 and c <=no_of_ones:
            bool_idx_list.append(True)
            c+=1
        elif y_test[idx] == 0 and c >no_of_ones:
            bool_idx_list.append(False)
        else:
            bool_idx_list.append(True)
    X_test = X_test[bool_idx_list]
    y_test = y_test[bool_idx_list]

    return X_test,y_test

def testing(seen_data=False,all_tasks=False):

    global training_cutoff

    dataset_loadtime=0
    
    task_CI_pnt = []
    test_CI_pnt =[]
    prauc_in_pnt = []
    prauc_out_pnt = []
    fpr_pnt = []   # FPR per task (FP / (FP + TN))
    fnr_pnt = []   # FNR per task (FN / (FN + TP))

    if seen_data:
        if all_tasks:
            testing_tasks = task_order
        else:
            testing_tasks = task_order[:training_cutoff]
        start_id = 0
    else:
        testing_tasks  = task_order[training_cutoff:]
        start_id = training_cutoff

    for task_id,task in enumerate(testing_tasks, start = start_id):
        
        task_class_ids = []
        task_minorityclass_ids = []
        
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        start = time.time()
        # input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=0,bool_encode_anomaly=1,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=True)
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=False,bool_encode_anomaly=False,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=False)
        
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
        
        val_pred, val_actual = [],[]
        for data, target in valid_loader: 
            with torch.no_grad():
                pred = torch.softmax(model(data.to(device)), dim=1)[:,1].reshape(target.shape)
            
            y_pred = pred.detach().cpu().numpy().tolist()

            val_pred.extend(y_pred)
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
        # precision, recall, thresholds = precision_recall_curve(val_actual, en_val_pred)
        # en_auc_precision_recall_1 = auc(recall, precision)

        precision, recall, thresholds = precision_recall_curve(val_actual, [1-val for val in val_pred], pos_label=0.)
        auc_precision_recall_0 = auc(recall, precision)
        # precision, recall, thresholds = precision_recall_curve(val_actual, [1-val for val in en_val_pred], pos_label=0.)
        # en_auc_precision_recall_0 = auc(recall, precision)

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

        # Compute FPR and FNR at threshold 0.5
        val_pred_binary = [1 if p >= 0.5 else 0 for p in val_pred]
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

    print('\npnt table:')
    pnt_table = [
        # ['task_CI']+ task_CI_pnt, 
        # ['test_CI'] + test_CI_pnt,
        ['prauc Benign traffic'] + prauc_in_pnt, 
        ['prauc Attack traffic'] + prauc_out_pnt
    ]
    print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
    
    print(f'dataset loading time: {dataset_loadtime}s\n')
    # indices: [0]=prauc_in_pnt, [1]=prauc_out_pnt, [2]=prauc_in_aut, [3]=prauc_out_aut,
    #          [4]=training_cutoff, [5]=seen_data, [6]=N, [7]=fpr_pnt, [8]=fnr_pnt
    return [prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N,fpr_pnt,fnr_pnt]

# def testing(seen_data=False):

#     global training_cutoff

#     dataset_loadtime=0
    
#     task_CI_pnt = []
#     test_CI_pnt =[]
#     prauc_in_pnt = []
#     prauc_out_pnt = []

#     if seen_data:
#         testing_tasks = task_order[:training_cutoff]
#         start_id = 0
#     else:
#         testing_tasks  = task_order[training_cutoff:]
#         start_id = training_cutoff

#     for task_id,task in enumerate(testing_tasks, start = start_id):
        
#         task_class_ids = []
#         task_minorityclass_ids = []
        
#         for class_ in task:
#             task_class_ids.extend([class_])
#             if class_ in minorityclass_ids:
#                 task_minorityclass_ids.extend([class_])
#         start = time.time()
#         # input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=0,bool_encode_anomaly=1,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=True)
#         input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=False,bool_encode_anomaly=False,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=False)
        
#         if seen_data:
#             features,target_label = X_test,y_test
#         else:
#             features,target_label = X_test,y_test
#             # features,target_label = tasks[0][0],tasks[0][1]
#         print(f'testing on task {task_id}: {features.shape}')
        
#         dataset_loadtime += time.time()-start
        
        
#         # valid_loader = torch.utils.data.DataLoader(dataset(tasks[0][0],tasks[0][1]),
#         #                                        batch_size=batch_size,
#         #                                     #    sampler=valid_sampler,
#         #                                        num_workers=0)     
#         valid_loader = torch.utils.data.DataLoader(dataset(features,target_label),
#                                                batch_size=batch_size,
#                                             #    sampler=valid_sampler,
#                                                num_workers=0)
        
#         val_pred, val_actual = [],[]
#         for data, target in valid_loader: 
#             with torch.no_grad():
#                 pred = torch.softmax(model(data.to(device)), dim=1)[:,1].reshape(target.shape)
            
#             y_pred = pred.detach().cpu().numpy().tolist()

#             val_pred.extend(y_pred)
#             val_actual.extend(target.detach().cpu().numpy().tolist())   
        
#         train_y = tasks[0][1]

#         # print(f'test set size:= {len(val_actual)} with {len([i for i in val_actual if int(i) == 1])} attacks')
#         # print(f'train set size:= {len(tasks[0][1])} with {len([i for i in tasks[0][1] if int(i) ==1])} attacks')
        
#         # train_CI = len([i for i in tasks[0][1] if round(i) == 1])/(len(tasks[0][1])-len([i for i in tasks[0][1] if round(i) == 1]))
#         # test_CI = len([i for i in val_actual if round(i) == 1])/(len(val_actual)-len([i for i in val_actual if round(i) == 1]))
#         # print(f'test CI: {test_CI}')
#         # print(f'train CI: {train_CI}')
        
#         precision, recall, thresholds = precision_recall_curve(val_actual, val_pred,pos_label=1.0)
#         auc_precision_recall_1 = auc(recall, precision)
#         # precision, recall, thresholds = precision_recall_curve(val_actual, en_val_pred)
#         # en_auc_precision_recall_1 = auc(recall, precision)

#         precision, recall, thresholds = precision_recall_curve(val_actual, [1-val for val in val_pred], pos_label=0.)
#         auc_precision_recall_0 = auc(recall, precision)
#         # precision, recall, thresholds = precision_recall_curve(val_actual, [1-val for val in en_val_pred], pos_label=0.)
#         # en_auc_precision_recall_0 = auc(recall, precision)

#         #when number of 1s > 0s then the 1 is the inliers and0 is the outliers
#         # auc_precision_recall_in = auc_precision_recall_0 if test_CI < 1 else auc_precision_recall_1
#         # auc_precision_recall_out = auc_precision_recall_1 if test_CI < 1 else auc_precision_recall_0

#         # en_auc_precision_recall_in = en_auc_precision_recall_0 if test_CI < 1 else en_auc_precision_recall_1
#         # en_auc_precision_recall_out = en_auc_precision_recall_1 if test_CI < 1 else en_auc_precision_recall_0

#         # task_CI_pnt.append(train_CI)
#         # test_CI_pnt.append(test_CI)
#         # prauc_in_pnt.append(auc_precision_recall_in)
#         # prauc_out_pnt.append(auc_precision_recall_out)
#         # en_prauc_in_pnt.append(en_auc_precision_recall_in)
#         # en_prauc_out_pnt.append(en_auc_precision_recall_out)
#         prauc_in_pnt.append(auc_precision_recall_0)
#         prauc_out_pnt.append(auc_precision_recall_1)
#         # en_prauc_in_pnt.append(en_auc_precision_recall_0)
#         # en_prauc_out_pnt.append(en_auc_precision_recall_1)
        
#         # print(f'prauc inliers: {auc_precision_recall_in}')        
#         # print(f'prauc outliers: {auc_precision_recall_out}')                     
#         # print('')
    
#     N = len(testing_tasks) #number of test tasks

#     if N<2:
#         print('not printing AUT values since it requires atleast 2 test tasks')
#         return
    
#     prauc_in_aut  = 0
#     prauc_out_aut = 0
#     for i in range(N-1):
#         prauc_in_aut+= (prauc_in_pnt[i]+prauc_in_pnt[i+1])/(2)
#         prauc_out_aut+=(prauc_out_pnt[i]+prauc_out_pnt[i+1])/(2)
#     prauc_in_aut  = prauc_in_aut/(N-1)
#     prauc_out_aut = prauc_out_aut/(N-1)
    
#     print(f'AUT(prauc inliers,{N}) := {prauc_in_aut}')
#     print(f'AUT(prauc outliers,{N}) := {prauc_out_aut}')

#     print('\npnt table:')
#     pnt_table = [
#         # ['task_CI']+ task_CI_pnt, 
#         # ['test_CI'] + test_CI_pnt,
#         ['prauc Benign traffic'] + prauc_in_pnt, 
#         ['prauc Attack traffic'] + prauc_out_pnt
#     ]
#     print(tabulate(pnt_table, headers = ['']+[str(training_cutoff+i) if not seen_data else str(i) for i in range(N)], tablefmt = 'grid'))
    
#     print(f'dataset loading time: {dataset_loadtime}s\n')
#     return [prauc_in_pnt,prauc_out_pnt,prauc_in_aut,prauc_out_aut,training_cutoff,seen_data,N]


def taskwise_lazytrain():
    global test_x,test_y
    # random.shuffle(task_order)
    for task_id,task in enumerate(task_order):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("loading task:",task_id)     
        input_shape,train_scenario = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=True,bool_encode_anomaly=False,label=label,bool_create_tasks_avalanche=True)
        test_x.extend([train_scenario[2][0]])
        test_y.extend([train_scenario[3][0]])
        print("Training task:",task_id)
        train_a_lazytask(train_scenario,task_label = [0,],task_id=task_id)
            # train(train_scenario=train_scenario,cl_strategy=get_cl_strategy()) 

    test_set_results = []
    auc_result= {}
    with open(temp_filename, 'w') as fp:
        # test_set_results.extend([testing(seen_data=True),testing(seen_data=False)])
        test_set_results.extend([testing(seen_data=True),testing(seen_data=False),testing(seen_data=True,all_tasks=True)])
        auc_result[str(args.seed)] = test_set_results
        json.dump(auc_result, fp)  
     

            
def evaluate_on_testset(X_test,y_test):

    X_test,y_test = avalanche_tensor_to_tensor(test_data_x = X_test,test_data_y = y_test)
    yhat = None
    model.eval()
    print("computing the results")
    offset = 250000
    for idx in range(0,X_test.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(X_test[idx1:idx2,:].astype(float)).to(device)
        
        # temp = model(X_test1.float()).detach().cpu().numpy()
        temp = model(X_test1.float()).detach().cpu().numpy()
        if idx1==0:
            yhat = temp
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
    compute_results(y_test,yhat)
    # print("test sample counters are",Counter(y_test))
    




def start_execution(dataset_name, seed,lr,w_decay):
    global input_shape,label,is_lazy_training,cl_strategy,test_x,test_y
    start_time=time.time()
    # load_model_metadata()
    load_metadata(dataset_name,lr,w_decay)
    if is_lazy_training:
        taskwise_lazytrain()
        
    else:
        input_shape,train_scenario = load_dataset(pth,class_ids,minorityclass_ids,tasks_list,task2_list,task_order,bool_encode_benign=False,bool_encode_anomaly=True,label=label,bool_create_tasks_avalanche=True)    
        test_x,test_y = train_scenario[2],train_scenario[3]
        train_scenario = create_avalanche_scenario(tasks_dataset=train_scenario,task_labels= [0 for key in range(0,no_tasks)])# using same variable to avoid multiple copies in the memory
        train(train_scenario=train_scenario,cl_strategy=cl_strategy)
       
    
    print("total training time is--- %s seconds ---" % (time.time() - start_time)) 
    print("seed is",seed)

    # evaluate_on_testset(test_x,test_y)

if __name__ == "__main__":
    import argparse

    global training_cutoff, temp_filename, cl_strat,lr,w_decay
    
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='random seed (default: 0)')
    parser.add_argument('--ds', type=str, default="ids18", metavar='S',help='dataset name')
    parser.add_argument('--training_cutoff', type=int, default=3, metavar='S',help='train the model for first n tasks and test for time decay on the rest')
    # parser.add_argument('--wd', type=float, default= 1e-03, metavar='S',help='optim weight decay')
    parser.add_argument('--lr', type=float, default=1e-02, metavar='S',help='batch memory ratio(default: 0.2)')
    parser.add_argument('--w_d', type=float, default=1e-4, metavar='S',help='labeled ratio (default: 0.1)')
    parser.add_argument('--cl_strat', type=int, default=0, metavar='S',help='integer id associated with the CL strategy)')
    parser.add_argument('--filename', type=str,default="temp", metavar='S',help='json file name')

    args = parser.parse_args()
    set_seed(args.seed)
    get_gpu(args.gpu)
    training_cutoff = args.training_cutoff
    # w_decay = float(args.wd)
    temp_filename = str(args.filename) 
    lr = float(args.lr)
    w_decay=float(args.w_d)
    cl_strat = args.cl_strat
    print("seed is",args.seed)

    print("{:<20}  {:<20}".format('Argument','Value'))
    print("*"*80)
    for arg in vars(args):
        print("{:<20}  {:<20}".format(arg, getattr(args, arg)))
    print("*"*80)

    start_execution(args.ds, args.seed,lr,w_decay)    
    