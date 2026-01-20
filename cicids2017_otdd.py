# prints Optimal Transport Dataset Distance (OTDD) for anoshift

import warnings
from datetime import datetime
import os
from math import ceil
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from otdd.pytorch.distance import DatasetDistance


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
start_time = time.time()
now = datetime.now()
cur_time = now.strftime("%d-%m-%Y::%H:%M:%S")

# Creating folder to save weights and logs
os.makedirs("weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)

train_data_x = []
train_data_y = []
test_data_x = []
test_data_y = []

names = ['0','10','11','12','20','21','22','23','24','25','30','31','32','33','34','35','40','41','42','43']#,'27','28']
minority_names = ['11','12','21','22','23','24','25','31','32','33','34','35','41','42','43']
pth = "../../../datasets/modified_cicids2017/ids17_day_wise/data/"
tasks_list = [0,10,20,30,40]
task2_list = ['0','10','20','30','40']
arrs,X_im,y_im,X_image = dict(),dict(),dict(),dict()
X_train, X_test, y_train, y_test = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
perm = {
    "1": [10,11,12],
    "2": [0,21],
    "3": [20,22,23,24,25],
    "4": [30,31,32,33,34,35],
    "5": [40,41,42,43],
    # "6":[20,7,32],
    # "7":[21,6,33],
    # "8":[34,22,10],
    # "9":[35,23,2],
    # "10":[36,24,13,1],
    # "11":[37,25,26]

}

#'task_order':[('15','14','12','90' ),('16','4','91', '9'),( '17','3','92', '5'),('18', '11','93'), ('94','19','8'),('20','7','95') ,('21','6','96' ),('97','22','10' ),('98','23','2' ),('99','24','13', '1' ),('100','25','26')]

for name in names:
  name1 = name+'.npy'
  print("loading "+pth+ str(name)+ ".npy file")
  data = np.load(pth+name1,allow_pickle=True) # loads the individual data for each class
  #data[:, -1], data[:, -2] = data[:, -2], data[:, -1].copy()
  #data[:,-1]=int(name)
  arrs[name] = data
  #print(arrs[name].shape)
  arr = arrs[name]
  X_im[name] = arr[:,:-1] #copying the features
  #print(X_im[name].shape)
  y_im[name] = arr[:,-1]  #copying the labels
  #print(y_im[name].shape)
  arr_X = X_im[name]
  data_image = []
  for j in range(arr_X.shape[0]):
    data_image.append(arr_X[j])
  list_to_array = np.array(data_image)
  X_image[name] = list_to_array 
  input_shape=data.shape[1]-1
total_minority_test_samples = 0
index_counter = {}
counter = 0
for i,name in enumerate(minority_names):
    arr_x = X_image[name]#df[df["Label"]==name]#X_image[name]
    arr_y = y_im[name]#arr_x.iloc[:,-1:]
    # arr_x.drop(arr_x.columns[len(arr_x.columns)-1], axis=1, inplace=True)
    class_length = arr_x.shape[0] # no.of instances per class
    
   
    train_split = int(0.75*class_length) 
    train_split2=int(0.05*class_length)   # I used 0.70 because I want 70% of the dataset to be train data, change this according to preference
    val_split = -int(0.002*class_length)  # Here I only want 0.2% of the dataset (around 6000 points) in my validation set, you can change this as well
                          # Whatever percentage is left forms the test dataset
   
    
    X_train = pd.concat([X_train, pd.DataFrame(arr_x[:train_split,:])], ignore_index=True)
    X_test = pd.concat([X_test, pd.DataFrame(arr_x[train_split:,:])], ignore_index=True)
    y_train = pd.concat([y_train, pd.DataFrame(arr_y[:train_split])], ignore_index=True)
    y_test = pd.concat([y_test, pd.DataFrame(arr_y[train_split:])], ignore_index=True)
    index_counter[name]= counter
    counter += 1
    
       
print(index_counter)       
    
total_minority_test_samples = y_test.shape[0]
print("total minority test samples",total_minority_test_samples)
samples_per_majority_class = ceil(total_minority_test_samples/len(perm))
majority_name = [class_ for class_ in names if class_ not in minority_names]

print("maj class",majority_name)
for i,name in enumerate(majority_name):
    arr_x = X_image[name]#df[df["Label"]==name]#X_image[name]
    arr_y = y_im[name]#arr_x.iloc[:,-1:]
    # arr_x.drop(arr_x.columns[len(arr_x.columns)-1], axis=1, inplace=True)
    class_length = arr_x.shape[0] # no.of instances per class    
    sample_ratio = (class_length-samples_per_majority_class)/class_length
    #print(class_length)
    train_split = int(0.75*class_length) 
    train_split2=int(0.05*class_length)   # I used 0.70 because I want 70% of the dataset to be train data, change this according to preference
    val_split = -int(0.002*class_length)  # Here I only want 0.2% of the dataset (around 6000 points) in my validation set, you can change this as well
                          # Whatever percentage is left forms the test dataset
                          # Whatever percentage is left forms the test dataset
    # X_train = X_train.append(pd.DataFrame(arr_x[:train_split,:]),ignore_index = True)
    # X_test = X_test.append(pd.DataFrame(arr_x[train_split:val_split,:]),ignore_index = True)
    # y_train = y_train.append(pd.DataFrame(arr_y[:train_split]),ignore_index = True)
    # y_test = y_test.append(pd.DataFrame(arr_y[train_split:val_split]),ignore_index = True)
    
    X_train = pd.concat([X_train, pd.DataFrame(arr_x[:train_split,:])], ignore_index=True)
    X_test = pd.concat([X_test, pd.DataFrame(arr_x[train_split:,:])], ignore_index=True)
    y_train = pd.concat([y_train, pd.DataFrame(arr_y[:train_split])], ignore_index=True)
    y_test = pd.concat([y_test, pd.DataFrame(arr_y[train_split:])], ignore_index=True)
    index_counter[name]= counter
    counter += 1

    # X_train, X_test, y_train, y_test = train_test_split(
    #     df, y, stratify=y, test_size=0.2498
    # )
print(index_counter)
print("total majority samples",(y_test.shape[0]-total_minority_test_samples))    


# del df
 

train_dict = {}
train_label_dict = {}
test_dict = {}
test_label_dict = {}

    # Labelling classses as 0 or 1 based on type of class.
    
print("y_train unique",y_train.nunique())  
print(y_test.nunique())   
for i in range(1,(y_train.iloc[:, -1].nunique()+1)):
    train_dict["cat" + str(i)] = X_train[y_train.iloc[:, -1] == i]

    temp = y_train[y_train.iloc[:, -1] == i]

        # Class label 0 = Normal class
    if i in tasks_list:
        temp.iloc[:, -1] = 0
    else:
        temp.iloc[:, -1] = 1

    train_label_dict["cat" + str(i)] = temp

for i in range(1,(y_test.iloc[:, -1].nunique()+1)):
    test_dict["cat" + str(i)] = X_test[y_test.iloc[:, -1] == i]

    temp = y_test[y_test.iloc[:, -1] == i]

    if i in tasks_list:
        temp.iloc[:, -1] = 0
    else:
        temp.iloc[:, -1] = 1

    test_label_dict["cat" + str(i)] = temp

train_data_x = list(torch.Tensor(
        train_dict[key].to_numpy()) for key in train_dict)
train_data_y = list(
        torch.Tensor(train_label_dict[key].to_numpy().flatten()) for key in train_label_dict
    )
test_data_x = list(torch.Tensor(
        test_dict[key].to_numpy()) for key in test_dict)
test_data_y = list(
        torch.Tensor(test_label_dict[key].to_numpy().flatten()) for key in test_label_dict
    )

    # with open("./data/train_data_x.pth", "wb") as f:
    #     pickle.dump(train_data_x, f)

    # with open("./data/train_data_y.pth", "wb") as f:
    #     pickle.dump(train_data_y, f)

    # with open("./data/test_data_x.pth", "wb") as f:
    #     pickle.dump(test_data_x, f)

    # with open("./data/test_data_y.pth", "wb") as f:
    #     pickle.dump(test_data_y, f)


def task_ordering(perm):
    """Divides Data into tasks based on the given permutation order

    Parameters
    ----------
    perm : dict
        Dictionary containing task id and the classes present in it.

    Returns
    -------
    tuple
        Final dataset divided into tasks
    """
    final_train_data_x = []
    final_train_data_y = []
    final_test_data_x = []
    final_test_data_y = []

    for key, values in perm.items():
        temp_train_data_x = torch.Tensor([])
        temp_train_data_y = torch.Tensor([])
        temp_test_data_x = torch.Tensor([])
        temp_test_data_y = torch.Tensor([])

        for value in values:
            print(key,value)
            
            temp_train_data_x = torch.cat([temp_train_data_x, train_data_x[value-1]])
            temp_train_data_y = torch.cat([temp_train_data_y, train_data_y[value-1]])
            temp_test_data_x = torch.cat([temp_test_data_x, test_data_x[value-1]])
            temp_test_data_y = torch.cat([temp_test_data_y, test_data_y[value-1]])

        final_train_data_x.append(temp_train_data_x)
        final_train_data_y.append(temp_train_data_y)
        final_test_data_x.append(temp_test_data_x)
        final_test_data_y.append(temp_test_data_y)

    final_train_data_y = [x.float() for x in final_train_data_y]
    final_test_data_y = [x.float() for x in final_test_data_y]
    
    return final_train_data_x, final_train_data_y, final_test_data_x, final_test_data_y

dataset = task_ordering(perm)



for i in range(1,len(dataset[0])-1):
    
    X = dataset[0][i]
    Y = dataset[1][i].to(torch.long)
    
    X2 = dataset[0][i+1]
    Y2 = dataset[1][i+1].to(torch.long)
    
    ds1 = TensorDataset(X, Y)
    ds1.classes = [0,1]
    ds2 = TensorDataset(X2, Y2)
    ds2.classes = [0,1]
    
    dist = DatasetDistance(ds1, ds2)
    print('(',i,',',i+1,') : ', dist.distance())   
    