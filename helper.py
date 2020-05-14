import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
import time

arch = {"vgg16": 25088,
        "densenet121": 1024,
        "alexnet": 9216}


def load_data(data_dir):
    '''
    arguments: data_dir (ex. ./flowers)
    returns: the dataloaders
    '''

    print("loading data...")
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # transformations
    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    # imageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # dataloader
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(
        valid_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=20, shuffle=True)

    print('data loaded!')
    return trainloader, validloader, testloader, train_data.class_to_idx


def nn_setup(structure, dropout, hidden_layer, lr, power):
    '''
    arguments: model structure, dropout probability, hidden layer nodes, learning rate, gpu
    returns: model structure, criterion, optimizer
    '''
    print("setting up neural network...")
    # checks which model network to use
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("{} is not a valid model. Choices: vgg16,densenet121,alexnet".format(
            structure))

    # model structure
    for param in model.parameters():
        param.requires_grad = False

        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(arch[structure], hidden_layer)),
            ('relu', nn.ReLU()),
            ('do1', nn.Dropout(dropout)),
            ('fc2', nn.Linear(hidden_layer, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr)

        if torch.cuda.is_available() and power == 'gpu':
            model.cuda()

        print("network set up!")
        return model, criterion, optimizer


def train_network(model, criterion, optimizer, epochs, print_every, trainloader, validloader, power):
    '''
    arguments: model, criterion, optimizer, epochs to run for, how many steps to print results, 
    the dataloader used to train, dataloader used for validation, gpu
    returns: nothing (trains the model and displays the losses+acciracy)
    '''
    # defines variables
    steps = 0
    running_loss = 0

    print('------------Training START------------')
    start_time = time.time()
    for e in range(epochs):
        running_loss = 0
        # training
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if torch.cuda.is_available() and power == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            # forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # test on validloader after every interval (print_every)
            if steps % print_every == 0:
                model.eval()
                # define variables
                vloss = 0
                accuracy = 0

                # validation
                for ii, (inputs2, labels2) in enumerate(validloader):
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to(
                            'cuda:0'), labels2.to('cuda:0')
                        model.to('cuda:0')

                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        vloss = criterion(outputs, labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(
                            torch.FloatTensor()).mean()

                vloss = vloss / len(validloader)
                accuracy = accuracy / len(validloader)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(vloss),
                      "Accuracy: {:.4f}".format(accuracy))

                running_loss = 0
        end_time = time.time()
        total_time = end_time - start_time
    print("-------------- Finished training-----------------------")
    print("----------Epochs: {}------------------------------------".format(epochs))
    print("----------Steps: {}-----------------------------".format(steps))
    print("----------Time: {}------------------------------".format(total_time))


def save_checkpoint(model, path, structure, hidden_layer, dropout, lr, class_to_idx):
    '''
    arguments: saving path, hyperparameters 
    (structure, hidden layer nodes, dropout, learning rate, epochs)
    returns: Nothing

    '''
    # model? from where... !
    print("saving checkpoint...")
    # model.cpu
    checkpoint = {'structure': structure,
                  'hidden_layer': hidden_layer,
                  'dropout': dropout,
                  'lr': lr,
                  'state_dict': model.state_dict(),
                  'class_to_idx': class_to_idx
                 }

    
    torch.save(checkpoint, path)
    print("checkpoint saved!")


def load_checkpoint(path, power):
    '''
    Arguments: The path of the checkpoint file
    Returns: The Neural Netowrk with all hyperparameters, weights and biases

    '''
    print(f"loading checkpoint... {path}")
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer = checkpoint['hidden_layer']
    dropout = checkpoint['dropout']
    lr = checkpoint['lr']
    
    model, criterion, optimizer = nn_setup(structure, dropout, hidden_layer, lr, power)

    model.load_state_dict(checkpoint['state_dict'])
    
    model.class_to_idx = checkpoint['class_to_idx']
    print("checkpoint loaded!")
    return model


def process_image(image_path):
    '''
    Arguments: The image's path
    Returns: The image as a tensor

    '''
    print("processing image...")

    # convert list to string if image_path is list

    img = Image.open(image_path)

    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    tensor_image = adjustments(img)

    print("image processed!")
    return tensor_image


def predict(image_path, model, topk, power):
    '''
    Arguments: The path to the image, the model, the number of prefictions and whether cuda will be used or not
    Returns: The "topk" most probable choices that the network predicts

    '''
    print("-----------Predicting----------------")
    if torch.cuda.is_available() and power == 'gpu':
        model.to('cuda:0')

    # process image, unsqueeze it, to float
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze(0)
    img_torch = img_torch.float()

    model.eval()
    
    if power == 'gpu':
        with torch.no_grad():
            print('using GPU')
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img_torch)

    probability = F.softmax(output.data, dim=1)

    print("------------Predictions finished!----------------")
    return probability.topk(topk), model.class_to_idx
