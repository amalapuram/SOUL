
from pickle import TRUE
import torch
import numpy as np
from torch.nn.utils import prune
from torch.nn.functional import cosine_similarity
from torchmetrics.functional import pairwise_cosine_similarity
import matplotlib.pyplot as plt
from numpy import dot
from torch.utils.data import TensorDataset
from numpy.linalg import norm
from utils.otdd.ot_distance import compute_ot_distance,compute_otdd_tabular_datasets
import torchvision
from sklearn import svm
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN
from sklearn.mixture import GaussianMixture

import random
import logging
# from imp import reload
# reload(logging)
import os
import time
import pprint
import math

from utils.config.configurations import cfg
from utils.classifiers import *
from utils.resnet import ResNet34,ResNet50,ResNet18,ResNetCLEAR,ResNetCLEAR50,ResNetCLEAR101
from torchvision.models import resnet101,resnet18,resnet50,googlenet,squeezenet1_0,ResNet18_Weights




# import numpy as np
# import torch


def clustering_class_imbalance(data):


    print("computing for DBScan")
    start_time = time.time()
    # Step 1: Apply DB Scan
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(data)
    print("total time",time.time()-start_time)

    print("computing for K-means")
    start_time = time.time()
    # Step 1: Apply K-Means
    kmeans = KMeans(n_clusters=2, random_state=42,n_init=100)
    kmeans_labels = kmeans.fit_predict(data)
    print("total time",time.time()-start_time)

    print("computing for GMM")
    start_time = time.time()
    # Step 2: Apply Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=2, random_state=42,covariance_type="diag")
    gmm_labels = gmm.fit_predict(data)
    print("total time",time.time()-start_time)
    
    # print("computing for Hierarchical clustering")
    # start_time = time.time()
    # # Step 3: Apply Hierarchical Clustering
    # hierarchical = AgglomerativeClustering(n_clusters=2)
    # hierarchical_labels = hierarchical.fit_predict(data)
    # print("total time",time.time()-start_time)

    # Step 4: Function to calculate imbalance ratio
    def calculate_imbalance_ratio(labels):
        class_counts = np.bincount(labels)
        if len(class_counts) < 2:
            return float('inf')  # If only one class is found
        majority_class = max(class_counts[0],class_counts[1])
        minority_class = min(class_counts[0],class_counts[1])
        imbalance_ratio = minority_class / (majority_class+minority_class) # if class_counts[1] > 0 else float('inf')
        return imbalance_ratio

    # Step 5: Calculate imbalance ratios
    dbscan_ratio = calculate_imbalance_ratio(dbscan_labels)
    print("DBScan CIR",dbscan_ratio)
    kmeans_ratio = calculate_imbalance_ratio(kmeans_labels)
    print("kmeans CIR",kmeans_ratio)
    gmm_ratio = calculate_imbalance_ratio(gmm_labels)
    print("GMM CIR",gmm_ratio)
    # hierarchical_ratio = calculate_imbalance_ratio(hierarchical_labels)
    # print("HC CIR",hierarchical_ratio)

    # Step 6: Calculate mean of the imbalance ratios
    # mean_imbalance_ratio = np.mean([kmeans_ratio, gmm_ratio])
    mean_imbalance_ratio = np.mean([kmeans_ratio, gmm_ratio, dbscan_ratio])

    return mean_imbalance_ratio

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0.01, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score1 = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss_min1 = np.Inf
        self.val_loss_min2 = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    #     score1 = val_loss[0]
    #     score2 = val_loss[1]

    #     if self.best_score1 is None:
    #         self.best_score1 = score1
    #         self.best_score2 = score2
    #         self.save_checkpoint(val_loss, model)
    #     elif score1 < self.best_score1 + self.delta and score2 > self.best_score2 + self.delta:
    #         self.counter += 1
    #         self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
    #         if self.counter >= self.patience:
    #             self.early_stop = True
    #     else:
    #         self.best_score1 = score1
    #         self.best_score2 = score2
    #         self.save_checkpoint(val_loss, model)
    #         self.counter = 0    

    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         self.trace_func(f'Validation PR-AUC (inliers) increased ({self.val_loss_min1:.6f} --> {val_loss[0]:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), self.path)
    #     self.val_loss_min1 = val_loss[0]
    #     self.val_loss_min2 = val_loss[1]


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation PR-AUC (inliers) increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class GradientRejection:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=1e-6, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self,  model):

        score = self.compute_gradientnorm(model)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
    
    def compute_gradientnorm(self,model):
        grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
        norm = torch.cat(grads).norm().detach().cpu().numpy().item()

        return norm
    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     self.trace_func(f'Validation PR-AUC (inliers) increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        # self.val_loss_min = val_loss        

def create_directories(label):
    output_root_dir = cfg.root_outputdir
    cl_strategy = cfg.clstrategy
    param_weights_dir_MIR = cfg.param_weights_dir_MIR
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    if not os.path.exists(param_weights_dir_MIR):
        os.makedirs(param_weights_dir_MIR)    
    timestamp = time.strftime("%d_%b_%H_%M_%S")  
    
    output_dir = output_root_dir +'/'+label+'/'+str(cl_strategy)+'/'+timestamp 
    if cfg.avalanche_dir:
        output_dir = output_root_dir +'/'+label+'/'+'avalanche'+'/'+str(cl_strategy)+'/'+timestamp
    cfg.outputdir = output_dir
    cfg.timestamp = timestamp 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + '/models')
        os.makedirs(output_dir + '/encoded_models')
        os.makedirs(output_dir + '/pickles')
        os.makedirs(output_dir + '/logs')
        os.makedirs(output_dir + '/plots')
        os.makedirs(output_dir + '/weights')

        # os.makedirs(output_dir + '')


def log(message, print_to_console=True, log_level=logging.DEBUG):
    if log_level == logging.INFO:
        logging.info(message)
    elif log_level == logging.DEBUG:
        logging.debug(message)
    elif log_level == logging.WARNING:
        logging.warning(message)
    elif log_level == logging.ERROR:
        logging.error(message)
    elif log_level == logging.CRITICAL:
        logging.critical(message)
    else:
        logging.debug(message)

    if print_to_console:
        print(message)             




def trigger_logging(label):
    output_root_dir = cfg.root_outputdir
    log_dir = output_root_dir+'/'+label+'/'+cfg.timestamp+'/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("time stamp is:",cfg.timestamp)    

    logging.basicConfig(filename=log_dir + '/'+cfg.timestamp + '.log', level=logging.DEBUG,force=True,
                        format='%(levelname)s:\t%(message)s')

    # log(pprint.pformat(cfg))    



def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n




def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    # torch.use_deterministic_algorithms(True)
    # torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset_info(dsname):
    if str(dsname) == "ids17":
        from utils.config.configurations import cicids2017 as ds   
    elif str(dsname) == "ids18":
        from utils.config.configurations import cicids2018 as ds
    elif str(dsname) == "kddcup99":
        from utils.config.configurations import kddcup99 as ds
    elif str(dsname) == "nslkdd":
        from utils.config.configurations import nslkdd as ds
    elif str(dsname) == "unswnb15":
        from utils.config.configurations import unswnb15 as ds    
    elif str(dsname) == "ctu13":
        from utils.config.configurations import ctu13 as ds    
    elif str(dsname) == "anoshift":
        from utils.config.configurations import anoshiftsubset as ds
    elif str(dsname) == "mnist":
        from utils.config.configurations import mnist as ds
    elif str(dsname) == "svhn":
        from utils.config.configurations import svhn as ds
    elif str(dsname) == "cifar10":
        from utils.config.configurations import cifar10 as ds
    elif str(dsname) == "cifar100":
        from utils.config.configurations import cifar100 as ds
    elif str(dsname) == "clear10":
        from utils.config.configurations import clear10 as ds 
    elif str(dsname) == "clear100":
        from utils.config.configurations import clear100 as ds  
    elif str(dsname) == "apigraph":
        from utils.config.configurations import apigraph as ds
    elif str(dsname) == "androzoo":
        from utils.config.configurations import androzoo as ds    
    return ds                          



def get_gpu(id):
    # gpu_list = cfg.gpu_ids.split(',')
    # print("gpu list",gpu_list)
    # gpus = [int(iter) for iter in gpu_list]
    cfg.device = torch.device('cuda:' + str(id)) 


model_path={'byol_imagenet':'./clear10/pretrain_weights/features/byol_imagenet/state_dict.pth.tar',
			'imagenet':	'./clear10/pretrain_weights/features/imagenet/state_dict.pth.tar',
			'moco_b0':'./clear10/pretrain_weights/features/moco_b0/state_dict.pth.tar',
			'moco_imagenet':'./clear10/pretrain_weights/features/moco_imagenet/state_dict.pth.tar'
            }   

def compute_cosine_sim(X,device):
    avg_cos_sim_vec = [0] * X.shape[0]
    threshold = 1000
    for idx in range(0,X.shape[0]):
        if X.shape[0] < threshold:
            sim = cosine_similarity(torch.from_numpy(X),torch.from_numpy(X[idx,:])).detach().cpu().numpy() 
        else:
            indicies = np.random.choice(X.shape[0],size = threshold,replace=False) 
            sim = cosine_similarity(torch.from_numpy(X[indicies,:]),torch.from_numpy(X[idx,:])).detach().cpu().numpy()    
        avg_cos_sim_vec[idx] = 1-np.average(sim)
        # print("Cosine si is",avg_cos_sim_vec[idx])

    return avg_cos_sim_vec    




def obtain_grad_vector(model,numpy_array):

    temp_list = []
    for param in model.parameters():
        temp_list.append(param.grad.view(-1))
    grads = torch.cat(temp_list).cpu().numpy().reshape(1,-1)
    
    if numpy_array is None:
        numpy_array = grads
    else:
        numpy_array = np.concatenate((numpy_array,grads), axis=0)    
 
    return numpy_array


def plot_cosine_sim(array1,array2,dir):
    cos_array = []
    # cos_array = cosine_similarity(torch.from_numpy(array1),torch.from_numpy(array2)).detach().cpu().numpy()
    for idx in range(0,array1.shape[0]):
        cos_sim = dot(array1[idx,:], array2[idx,:])/((norm(array1[idx,:])*norm(array2[idx,:])))
        cos_array.append(cos_sim)
        # print(cos_sim)
    os.makedirs(dir,exist_ok=True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis('on')
    # matplotlib.rcParams.update({'font.size': 18})
    plt.legend(prop={'size': 18})
    plt.title("Cosine Similarity bw gradients") 
    plt.xlabel("Batch num")
    # plt.yticks(np.arange(-0.002, 1, 0.02))
    plt.ylabel("Cosine similarity")
    plt.figure(figsize=(10,6))
    
    plt.plot(range(0,len(cos_array)), cos_array, color ="red")
    plt.savefig(dir+'/'+'cosine_sim_grads.pdf')
    # plt.show()


def check_grad_exist(model):

    for param in model.parameters():
        if param.grad is None:
            return False 


    return True

def dataset_from_numpy(X, Y, classes = None):
    targets =  torch.LongTensor(list(Y))
    ds = TensorDataset(torch.from_numpy(X).type(torch.FloatTensor),targets)
    ds.targets =  targets
    ds.classes = classes if classes is not None else [i for i in range(len(np.unique(Y)))]
    return ds

def compute_otdd_bwtasks(task1,task2,classlabel):
    
    transform=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.Grayscale(3),torchvision.transforms.ToTensor()])
    transform2=torchvision.transforms.Compose([torchvision.transforms.Resize([28,28])])

    for a,b,_ in task1:
        x1,y1 = a,b
        print(x1.shape[0],y1.shape[0])
        
    for a,b,_ in task2:
        x2,y2 = a,b  
    
    # for t1,t2,_,t3,t4,_ in zip((task1,task2)):
    #         x1,y1,x2,y2 = t1,t2,t3,t4
    
    if classlabel == "cifar10":
        x1,y1,x2,y2 = x1.reshape(-1,3,32,32),y1,x2.reshape(-1,3,32,32),y2
        return compute_ot_distance(x1,y1,x2,y2,device="cpu",bool_feature_cost="mnist")
    
    elif classlabel == "mnist_cifar10":
        x_temp = np.zeros((x1.shape[0],3,28,28))
        x_temp2 = np.zeros((x2.shape[0],3,28,28))
        for i in range(0,x1.shape[0]):  
            if  x1[i,:].shape[0] == 784:
                x_temp[i,:,:,:] = transform((x1[i,:].reshape(28,28,1))).numpy()
            else:
                x_temp[i,:,:,:] = transform2(torch.from_numpy(x1[i,:].reshape(3,32,32))).numpy() 

        for i in range(0,x2.shape[0]):  
            if x2[i,:].shape[0] != 784:    
                x_temp2[i,:,:,:] = transform2(torch.from_numpy(x2[i,:].reshape(3,32,32))).numpy()  
            else:
                x_temp2[i,:,:,:] = transform((x2[i,:].reshape(28,28,1))).numpy()

        return compute_ot_distance(x_temp,y1,x_temp2,y2,device="cpu",bool_feature_cost="mnist_cifar10")  
    
    elif classlabel == "mnist":
        x_temp = np.zeros((x1.shape[0],3,28,28))
        x_temp2 = np.zeros((x2.shape[0],3,28,28))
        for i in range(0,x1.shape[0]):  
            x_temp[i,:,:,:] = transform((x1[i,:].reshape(28,28,1))).numpy()
            
        for i in range(0,x2.shape[0]):  
            x_temp2[i,:,:,:] = transform((x2[i,:].reshape(28,28,1))).numpy()

        return compute_ot_distance(x_temp,y1,x_temp2,y2,device="cpu",bool_feature_cost="mnist")   
    else:
        ds1 = dataset_from_numpy(x1,y1)
        # ds1.classes = [0,1]
        ds2 = dataset_from_numpy(x2, y2)
        # ds2.classes = [0,1] 
        
        return compute_otdd_tabular_datasets(ds1,ds2) 







def prune_model_global_unstructured(model, layer_type, proportion):
    module_tups = []
    for module in model.modules():
        if isinstance(module, layer_type):
            module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=proportion
    )
    for module, _ in module_tups:
        prune.remove(module, 'weight')
    return model             

def extract_features_with_resnet18(X,device):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    offset = 100
    for idx in range(0,X.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(X[idx1:idx2,:]).to(device)#.astype(float))#.to(device)
        temp = feature_extractor(X_test1.float()).detach().cpu().numpy() 
        X_test1.detach()
        del X_test1
        if idx1==0:
            X_features = temp
        else:
             X_features = np.append( X_features, np.array(temp), axis=0)
    with torch.no_grad():
        # temp_model.detach()
        torch.cuda.empty_cache()
        del feature_extractor
    return  X_features     






def load_model_clear10(state_dict_path):
    # model=resnet50(pretrained=False)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    n_inputs = model.fc.in_features
    model.fc=torch.nn.Identity()
    # state_dict=torch.load(state_dict_path)
    # model.load_state_dict(state_dict)
    # for p in model.parameters():
    #     p.requires_grad= False
    
    layers = []
    layers.append(nn.Linear(n_inputs,1))
    layers.append(nn.Sigmoid())   
    model.fc = nn.Sequential(*layers)
	
    return model  

def load_googlenetmodel_clear10(state_dict_path):
    # model=resnet50(pretrained=False)
    model = googlenet(pretrained=True)
    n_inputs = model.fc.in_features
    model.fc=torch.nn.Identity()
    # state_dict=torch.load(state_dict_path)
    # model.load_state_dict(state_dict)
    # for p in model.parameters():
    #     p.requires_grad= False
    
    layers = []
    layers.append(nn.Linear(n_inputs,1))
    layers.append(nn.Sigmoid())   
    model.fc = nn.Sequential(*layers)
	#model.fc=torch.nn.Identity()
    
    
	# model.eval()
    return model  
    
def load_squeezenetmodel_clear10(state_dict_path):
    # model=resnet50(pretrained=False)
    model = squeezenet1_0(pretrained=True)
    model.num_classes = 1
    final_conv = nn.Conv2d(512, model.num_classes, kernel_size=1)
    model.classifier = nn.Sequential(
            nn.Dropout(p=0.5), final_conv, nn.Sigmoid(), nn.AdaptiveAvgPool2d((1, 1))
        )
# change the internal num_classes variable rather than redefining the forward pass
    
    # state_dict=torch.load(state_dict_path)
    # model.load_state_dict(state_dict)
    # for p in model.parameters():
    #     p.requires_grad= False
    
    
    
    
	# model.eval()
    return model             




def load_model(label,inputsize,softmax=False):
    model = None
    if label == 'cicids2017':
         
         model = CICIDS2017_FC(inputsize=inputsize, softmax=softmax)
        #model = CICIDS2018_FC(inputsize=inputsize, softmax=softmax)

        
        # model = CICIDS2017_RF_MLP(inputsize=inputsize,num_experts=1,keep_prob=0.9,softmax=softmax)
        
    elif label ==  'cicids2018':
        model = CICIDS2018_FC(inputsize=inputsize, softmax=softmax)
    elif label == 'unswnb15':
        model = UNSWNB15_FC(inputsize=inputsize)  
    elif label == 'anoshift_subset':
        model = ANOSHIFT_FC(inputsize=inputsize, softmax=softmax)
    elif label == 'anoshift_subset_supervised':
        model = ANOSHIFT_FC_SUPERVISED(inputsize=inputsize, softmax=softmax)    
    elif label == 'anoshift_subset_student2':
        model = ANOSHIFT_FC_STUDENT2(inputsize=inputsize, softmax=softmax)   
    elif label == 'androzoo':
        model = ANDROZOO_FC(inputsize=inputsize, softmax=softmax)
    elif label == 'apigraph':
        model = APIGRAPH_FC(inputsize=inputsize, softmax=softmax)
    elif label == 'ctu13':
        model = CTU13_FC(inputsize=inputsize)    
    elif label == 'mnist':
        model = MNIST_FC(inputsize=inputsize)  
    elif label == 'svhn':
        # model =  SVHN_FC(inputsize=inputsize)   
        model = resnet18(pretrained=True)
        n_inputs = model.fc.in_features
        layers = []
        # layers.append(nn.Linear(n_inputs,100))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(100,50))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(50,2))
        # layers.append(nn.Sigmoid())
        # layers.append(torch.nn.Softmax(dim=1))
        layers.append(nn.Linear(n_inputs,1))
        # # layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())   
        model.fc = nn.Sequential(*layers)
    elif label == "mnist_cifar10":
        model = resnet18(pretrained=True)
        n_inputs = model.fc.in_features
        layers = []
        layers.append(nn.Linear(n_inputs,100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100,50))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(50,1))
        layers.append(nn.Sigmoid())
        # layers.append(torch.nn.Softmax(dim=1))
        # layers.append(nn.Linear(n_inputs,1))
        # # layers.append(nn.ReLU())
        # layers.append(nn.Sigmoid())   
        model.fc = nn.Sequential(*layers)
    elif label in ['cifar100','cifar100_large_benign']:
        model = resnet18(pretrained=True)
        n_inputs = model.fc.in_features
        layers = []
        layers = []
        layers.append(nn.Linear(n_inputs,100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100,50))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(50,1))
        layers.append(nn.Sigmoid())   
        model.fc = nn.Sequential(*layers)    
    elif label == 'clear100':
        model = resnet18(pretrained=True)
        n_inputs = model.fc.in_features
        layers = []
        layers = []
        layers.append(nn.Linear(n_inputs,100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100,50))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(50,1))
        layers.append(nn.Sigmoid())   
        model.fc = nn.Sequential(*layers)
    elif label == 'cifar10':
        model = resnet18(pretrained=True)
        n_inputs = model.fc.in_features
        layers = []
        layers.append(nn.Linear(n_inputs,100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100,50))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(50,1))
        layers.append(nn.Sigmoid())
        # layers.append(torch.nn.Softmax(dim=1))
        # layers.append(nn.Linear(n_inputs,1))
        # # layers.append(nn.ReLU())
        # layers.append(nn.Sigmoid())   
        model.fc = nn.Sequential(*layers)
     
    elif label in ['kddcup99','nslkdd']:
        model = KDDCUP99_FC(inputsize=inputsize) 
    elif label == 'clear10':        
        model = resnet18(pretrained=True)
        n_inputs = model.fc.in_features
        layers = []
        layers.append(nn.Linear(n_inputs,100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100,50))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(50,1))
        layers.append(nn.Sigmoid())
        # layers.append(torch.nn.Softmax(dim=1))
        # layers.append(nn.Linear(n_inputs,1))
        # # layers.append(nn.ReLU())
        # layers.append(nn.Sigmoid())   
        model.fc = nn.Sequential(*layers)

        # model = ResNetCLEAR()
    elif label in ['MSL','SMD','SMAP']:
        model=SMAP_FC(inputsize=inputsize)   
    else:
        model = CIDDS_FC(inputsize=inputsize)   

    return model    


def load_LSTM(input_size, hidden_size, num_layers,output_size):
    return custom_LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,output_size=output_size)   

def load_weightsFC(inputsize):
    return Weights_FC(inputsize)        

