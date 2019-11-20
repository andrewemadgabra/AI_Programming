import argparse
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from os.path import isdir
from torchvision import datasets, transforms, models




def arg_parser():
    
    parser = argparse.ArgumentParser(description="neural network")
    
    parser.add_argument('--arch',
                        action="store",
                        default="vgg16",
                        type=str)
    
    parser.add_argument('--dir', 
                        type=str,nargs='*',
                        action="store",
                        default="./flowers/")
    
   
    parser.add_argument('--rate', 
                        action="store",
                        default=0.001,
                        type=float)
    
    parser.add_argument('--units',
                        action="store", 
                        default=25088,
                        type=int)
    
    parser.add_argument('--epochs',
                        action="store",
                        default=1,
                        type=int)

    parser.add_argument('--gpu', 
                        default="gpu",
                        action="store_true")
    
    
    args = parser.parse_args()
    return args

#function transform and load data
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

def network_layer_setup_model(arch='vgg16',hidden_units=5120,learning_rate=0.001):
    model =  getattr(models,arch)(pretrained=True)
    in_features = model.classifier[0].in_features
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                           ('fc1',nn.Linear(in_features,hidden_units)),
                           ('ReLu1',nn.ReLU()),
                           ('Dropout1',nn.Dropout(p=0.15)),
                           ('fc2',nn.Linear(hidden_units,512)),
                           ('ReLu2',nn.ReLU()),
                           ('Dropout2',nn.Dropout(p=0.15)),
                           ('fc3',nn.Linear(512,102)),
                           ('output',nn.LogSoftmax(dim=1))
                           ]))

    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)

    return  criterion, optimizer , model


# Validation: accuracy and validation set loss
def valid(criterion, model, validloader):
    accuracy = 0
    val_loss = 0
    model.to('cuda')

    with torch.no_grad():
        for data in validloader:
            images, labels = data
            if gpu==True:
                images, labels = images.to('cuda'), labels.to('cuda')
                
            for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    output = model.forward(images)
                    val_loss += criterion(output, labels).item()
                    ps = torch.exp(output)
                    equality = (labels.data == ps.max(dim=1)[1])
                    equality.cpu()
                    accuracy += equality.type_as(torch.FloatTensor).mean()    
        
    return val_loss, accuracy

def network_learn(model, criterion, optimizer, epochs, print_result, loader, power):
    step = 0
    delay = 0

    for e in range(epochs):
        delay = 0
        for ii, (inputs, labels) in enumerate(loader):
            step += 1
            if torch.cuda.is_available() and power == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            delay += loss.item()

            if step % print_result == 0:
                model.eval()
                vlost = 0
                accuracy=0

                with torch.no_grad():
                    vlost, accuracy = valid(criterion,model,loader)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_result
                      ),
                      "Validat Lost {:.4f}".format(vlost / len(vloader)),
                       "Accuracy: {:.4f}".format(accuracy /len(vloader)))
                delay = 0
                model.train()


                               
def save_model(model_trained):

    model_trained.class_to_idx = image_datasets['train'].class_to_idx
    model_trained.cpu()
    save_dir = ''
    checkpoint = {
             'arch': arch,
             'units': units, 
             'state_dict': model_trained.state_dict(),
             'class_to_idx': model_trained.class_to_idx,
             }
    network_layer_setup_model(checkpoint)
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = 'check.pth'

    torch.save(checkpoint, 'check.pth')
    
def main():
    input = arg_parser()
    
    
    if input.arch == 'vgg16':
        input_size = 25088
        model = models.vgg16(pretrained=True)
    elif input.arch == 'densenet':
        model=models.densenet121(pretrained=True)
        input_size=1024

    print_every_time = 30
    steps = 0
    epochs = 3
    
    
    train_load, v_loader, test_load = dataload(input.dir)
    criterion, optimizer , model = network_layer_setup_model(input.arch, input.units, input.rate)
    model_trained = network_learn(model,criterion,optimizer,epochs,print_every_time,train_load,input.gpu)
    save_model(model_trained)


if __name__ == '__main__': main()
