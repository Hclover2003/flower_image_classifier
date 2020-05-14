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

import helper


ap = argparse.ArgumentParser(description='Predict.py')

# Command line arguments
# 5 of them: image to be classified, checkpoint file location, cat_to_name file, gpu, topk
ap.add_argument('input_img', type=str,
                default='/home/workspace/ImageClassifier/flowers/test/1/image_06752.jpg')
ap.add_argument('checkpoint', type=str,
                default='/home/workspace/ImageClassifier/checkpoint.pth')
ap.add_argument('--category_names', dest="category_names",
                type=str, default='cat_to_name.json')
ap.add_argument('--gpu', dest="gpu", type=str, default="gpu")
ap.add_argument('--top_k', dest="top_k", type=int, default=5)


pa = ap.parse_args()

print(pa.input_img)

input_img = pa.input_img
path = pa.checkpoint
category_names = pa.category_names
power = pa.gpu
number_of_outputs = pa.top_k


# load trained model
model = helper.load_checkpoint(path, power)

probabilities, class_to_idx = helper.predict(input_img, model, number_of_outputs, power)

with open(category_names, 'r') as json_file:
    cat_to_name = json.load(json_file)

    
classes = []     
for item in np.array(probabilities[1][0]):
    for k, v in class_to_idx.items():
        if item == v:
            classes.append(k)
            break
                

labels = [cat_to_name[str(index)] for index in classes]
predictions = np.array(probabilities[0][0])

i = 0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], predictions[i]))
    i += 1

print("-----------Prediction Complete-------")
