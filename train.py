import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import helper

ap = argparse.ArgumentParser(description='Train.py')

# Command line arguments
# 8 of them: data_dir, save_dir, arch, gpu, epochs, hidden_units, learning_rate, dropout

ap.add_argument('data_dir', type=str, default="./flowers/")
ap.add_argument('--save_dir', type=str, dest="save_dir",
                default="./checkpoint.pth")
ap.add_argument('--arch', type=str, dest="arch", default="vgg16")
ap.add_argument('--gpu', type=str, dest="gpu", default="gpu")
ap.add_argument('--epochs', type=int,  dest="epochs", default=1)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", default=4096)
ap.add_argument('--learning_rate', type=float,
                dest="learning_rate", default=0.001)
ap.add_argument('--dropout', type=float, dest="dropout", default=0.5)

pa = ap.parse_args()

data_dir = pa.data_dir
path = pa.save_dir
structure = pa.arch
power = pa.gpu
epochs = pa.epochs
hidden_layer = pa.hidden_units
lr = pa.learning_rate
dropout = pa.dropout


# load the 3 dataloaders (takes data_dir as argument)
trainloader, validloader, testloader, class_to_idx = helper.load_data(data_dir)

# set up model structure (takes structure, dropout, hidden_layer, lr, power)
model, optimizer, criterion = helper.nn_setup(
    structure, dropout, hidden_layer, lr, power)

# train and validate the network
helper.train_network(model, optimizer, criterion,
                     epochs, 20, trainloader, validloader, power)


helper.save_checkpoint(model, path, structure,
                       hidden_layer, dropout, lr, class_to_idx)


print("------------Model Trained!------------")
