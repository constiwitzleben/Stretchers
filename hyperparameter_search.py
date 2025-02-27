import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from util.Affine_Transformations import generate_strain_tensors, generate_27_strain_tensors
import matplotlib.pyplot as plt
from training_models import SuperNet, TripleNet
from training import train

file_path = "data/SuperPoint_Descriptors_Dataset.pth"
data = torch.load(file_path)
base_descriptors = data['descriptors']
# base_descriptors = base_descriptors / torch.norm(base_descriptors,dim=-1,keepdim=True)
transformed_descriptors = data['deformed_descriptors']
# transformed_descriptors = transformed_descriptors / torch.norm(transformed_descriptors,dim=-1,keepdim=True)
transformation_idxs = data['transformations']
transformation_idxs = torch.repeat_interleave(transformation_idxs, repeats=14, dim=1).flatten()
transformation_list = np.array(generate_strain_tensors())
parameters = torch.tensor(transformation_list[transformation_idxs])

dataset = TensorDataset(base_descriptors, transformed_descriptors, parameters)
samples = len(dataset)
train_size = int(0.8*samples)
val_size = int(0.2*samples)
train_dataset = Subset(dataset, list(range(train_size)))
val_dataset = Subset(dataset, list(range(train_size, samples)))


def objective(trial):
    device = torch.device('cpu')

    num_layers = trial.suggest_int('num_layers',0,3)
    hidden_dimension = trial.suggest_categorical('hidden_dimension',[64,256,512,1024,2048])
    lr = trial.suggest_float('lr',1e-6,1e-2,log=True)
    batch_size = trial.suggest_categorical('batch_size',[16,32,64,128,256])
    model_type = trial.suggest_categorical('model_type', ['SuperNet', 'TripleNet'])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = False)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last = False)

    if model_type == 'SuperNet':
        model = SuperNet(256,3,hidden_dimension,num_layers).to(torch.float32).to(device)
    if model_type == 'TripleNet':
        model = TripleNet(256,3,hidden_dimension,num_layers).to(torch.float32).to(device)

    
    optimizer = optimizer = optim.Adam(model.parameters(), lr=lr)

    flag = f'{model_type}-{num_layers}-{hidden_dimension}-lr:{lr}-bs:{batch_size}'

    train(model,train_dataloader,val_dataloader,optimizer,8,device,flag)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 100, n_jobs = 20)



    
