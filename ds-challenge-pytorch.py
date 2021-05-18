import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import eli5
'''
This function builds a predictive model to predict whether a vessel will be involved in an incident in the next 12 months. 
The predictor is a neural network with a logistic regression acting as a benchmark 
PRE: Takes as argument 
  reg1: The Lasso regularisatin parameter to train a neural network 
  reg2: The Ridge regularization paramter to train a neural network
  lr  : The learning rate of the neural network 
  
POST: 	The method will pre process the input data ./nav_inc_data.pkl, dropping null fields and train a neural network and a logistic regression on this data. 
  The feature detection uses the Mean Decrease Accuracy algorithm. 
'''
marker = "#"*10
# import and do a sanity check of the data
print(marker, "Loading and inspecting the data...")
ds = pd.read_pickle("/home/macenrola/Documents/siriusinsightai/ds_challenge/nav_inc_data.pkl")
# print(ds.head()) 
# print(ds.info())
# print(sum(ds["nav_incident"])) 

    
# Normalization
print(marker,"Removal of undesired fields")
columns_to_drop = ['enc_flag', 'enc_sreg', 'enc_cport', 'Management Risk', 'Owner Risk', 'Previous Years Number of Claims', 
          'Previous Years Number of Claims of Interest',  'safety_deficiencies_oDay'] 
print(marker,"Dropping the following fields: ", columns_to_drop)
ds_normalized = ds.drop(columns=columns_to_drop).copy(True) # removing the columns that don't contain useful information and those with no nonzero entries

print(marker,"Data normalization, zero centering and division by the standard deviation of each field")
ds_normalized.iloc[:,0:-1] = ds_normalized.iloc[:,0:-1].apply(lambda x: (x-x.mean())/ x.std(), axis=0) # so we normalize them to have zero mean and std deviation of 1
print(ds_normalized.info())



# Training
# split in training, optimization and validation datasets
split = 0.2
print(marker,"Splitting the data in training ({}%) and validation ({}%) datasets".format(100*(1-split), 100*split))
ds_normalized = ds_normalized.sample(frac=1).reset_index(drop=True)
n_train, n_eval = int(np.floor(len(ds_normalized)*(1-split))), int(np.floor(len(ds_normalized)*split))

X_train, y_train  = ds_normalized.iloc[:n_train, 0:-1], ds_normalized.iloc[:n_train,   -1]
X_eval, y_eval = ds_normalized.iloc[n_train:n_train+n_eval, 0:-1], ds_normalized.iloc[n_train:n_train+n_eval,   -1]
X_all, y_all =  ds_normalized.iloc[:,0:-1], ds_normalized.iloc[:,-1]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

N, input_size = X_train.shape
device = torch.device('cpu')

class BoatieDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels):
        'Initialization'
        self.labels = labels
        self.features = features

  def __len__(self):
        'Denotes the total number of samples'
        return self.features.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.features.iloc[index, :]
        y = self.labels.iloc[index]

        return torch.FloatTensor(X), y

class BoatieNet(nn.Module):

    def __init__(self):
        super(BoatieNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
       
    def forward(self, x):
        x = self.linear_relu_stack(x).flatten()
        return x


training_data = BoatieDataset(X_train, y_train)
test_data     = BoatieDataset(X_eval, y_eval)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

net = BoatieNet()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

size = len(train_dataloader.dataset)

def train(dataloader, model, loss_fn, optimizer):
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.float(), y.float())
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred.float(), y.float()).item()
            correct += (torch.round(pred.float()) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, net, loss_fn, optimizer)
    test(test_dataloader, net)
print("Done!")
