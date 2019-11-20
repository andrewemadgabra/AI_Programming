import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image
from os import listdir
import json
import argparse


def dataload(data_dir):
  train_transforms = transforms.Compose([transforms.RandomRotation(30),
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

    # Load the Data
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32,shuffle = True)
    
    return trainloader, testloader, validloader




def load_file(path='check.pth'):
    
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    units_layer = checkpoint['units']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']

    model,_,_ = nn_setup(structure , dropout,units_layer,lr)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

def process_image(image_path):
    
    for i in image_path:
        path = str(i)
    img = Image.open(i)

    make_img= transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_image = make_img(img)

    return tensor_image

def predict(image_path, model, topk=5):

    img = Image.open(image_path)
    image = process_image(img)
    output = model(image)
    probs, indices = output.topk(topk)
    
    index = {val: key for key, val in cat_to_name.items()} 
    top_level = [index_to_class[each] for each in indices]
    
    return probs, top_level




def main():
    input = argparse.ArgumentParser(description='predict-file')
    input.add_argument('input_test', default='paind-project/flowers/test/100/image_07896.jpg', nargs='*', action="store", type = str)
    input.add_argument('point', default='/home/workspace/ImageClassifier/checkpoint.pth', nargs='*', action="store",type = str)
    input.add_argument('--top_k', default=5, action="store", type=int)
    input.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    input.add_argument('--gpu', default="gpu", action="store" )
    input.add_argument('--dir', type=str,nargs='*',action="store",default="./flowers/")
    input.add_argument("--json_file", type=str, help="path file that maps the class values to other category names", default='cat_to_name.json')

    passes = input.arg_parser()
    path_image = passes.input_test
    number_of_outputs = passes.top_k
    power = passes.gpu
    input_img = passes.input_img
    data_path = passes.dir
    path_class = passes.json_file

    trainloader, testloader, validloader = dataload(data_path)


    with open(path_class, 'r') as data_read:
        cat_to_name = json.load(data_read)
    

    probs, classes = predict(image_path, model, topk)
    sb.countplot(y = classes, x = probs, color ='blue', ecolor='black', align='center')
    
    plt.show()
    ax.imsow(image)

if __name__ == '__main__': main()
