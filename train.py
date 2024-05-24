import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from torch import nn,optim
from torchvision import datasets,transforms,models
from collections import OrderedDict
from PIL import Image
import argparse

def get_input_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('dir',type=str,
                       help='path to the folder of images')
    parser.add_argument('--save_dir',type=str,default='model_dir',
                       help='path to the saved folder')
    parser.add_argument('--arch',type=str,default='densenet121',
                       help='name of the architecture')
    parser.add_argument('--learning_rate',type=float,default=0.001,
                       help='learining rate of the model')
    parser.add_argument('--epochs',type=int,default=5,
                       help='Number of iterations to train on train data')
    parser.add_argument('--hidden_units',type=int,default=512,
                       help='Number of iterations to train on train data')
    parser.add_argument('--gpu',action='store_true',
                       help='whether to use gpu or not')
    return parser.parse_args()

def augmentation(path):
    train_dir=path+'/train'
    valid_dir=path+'/valid'
    test_dir=path+'/test'
    data_transforms = [transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(30),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                     ]),
                  transforms.Compose([transforms.Resize(254),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
                                    )]

    # Load the datasets with ImageFolder
    image_datasets = [datasets.ImageFolder(train_dir,transform=data_transforms[0]),
                      datasets.ImageFolder(valid_dir,transform=data_transforms[1]),
                     datasets.ImageFolder(test_dir,transform=data_transforms[1])]

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0],batch_size=64,shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1],batch_size=64,shuffle=True),
                  torch.utils.data.DataLoader(image_datasets[2],batch_size=64,shuffle=True),]
    return dataloaders,image_datasets

def define_model(mode,hiddenunits):
    if mode in 'vgg13':
        model=models.vgg13(pretrained=True)
        output=25088
    elif mode in 'vgg16':
        model=models.vgg16(pretrained=True)
        output=25088
    elif mode in 'densenet121':
        model=models.densenet121(pretrained=True)
        output=1024
    elif mode in 'densenet161':
        model=models.densenet161(pretrained=True)
        output=2208
    elif mode in 'densenet201':
        model=models.densenet201(pretrained=True)
        output=1920
    elif mode in 'resnet50':
        model=models.resnet50(pretrained=True)
        output=2048
    
    for param in model.parameters():
        param.requires_grad=False
        
    model.classifier=nn.Sequential(OrderedDict([
        ('fc1',nn.Linear(output,hiddenunits)),
        ('relu1',nn.ReLU()),
        ('drop1',nn.Dropout(p=0.2)),
        ('fc2',nn.Linear(512,256)),
        ('relu1',nn.ReLU()),
        ('fc3',nn.Linear(256,102)),
        ('output',nn.LogSoftmax(dim=1))
    ]))
    return model



def main():
    in_arg=get_input_args()
    dataloaders,image_datasets=augmentation(in_arg.dir)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    model=define_model(in_arg.arch,in_arg.hidden_units)
    criterion=nn.NLLLoss()

    optimizer=optim.Adam(model.classifier.parameters(),lr=in_arg.learning_rate)

    if in_arg.gpu:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device="cpu"
    model.to(device)
    epochs=in_arg.epochs
    steps=0
    run_loss=0
    for_every=10
    for epoch in range(epochs):
        for images,labels in dataloaders[0]:
            steps+=1
            images,labels=images.to(device),labels.to(device)

            optimizer.zero_grad()

            ps=model.forward(images)
            loss=criterion(ps,labels)
            loss.backward()
            optimizer.step()

            run_loss+=loss.item()
            if steps%for_every==0:
                validation_loss=0
                accuracy=0
                model.eval()
                with torch.no_grad():
                    for images,labels in dataloaders[1]:
                        images,labels=images.to(device),labels.to(device)

                        ps=model.forward(images)
                        batch_loss=criterion(ps,labels)

                        validation_loss+=batch_loss.item()

                        out=torch.exp(ps)
                        top_p,top_label=out.topk(1,dim=1)
                        equals=top_label==labels.view(*top_label.shape)

                        accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}..."
                     f"Train loss: {run_loss/for_every:.3f}..."
                     f"Validation loss: {validation_loss/len(dataloaders[1]):.3f}..."
                     f"Validation Accuracy: {accuracy/len(dataloaders[1]):.3f}")
                run_loss=0 
                model.train()
    test_loss=0
    accuracy=0
    model.eval()
    with torch.no_grad():
        for images,labels in dataloaders[2]:
            images,labels=images.to(device),labels.to(device)

            ps=model.forward(images)
            batch_loss=criterion(ps,labels)

            test_loss+=batch_loss.item()

            out=torch.exp(ps)
            top_p,top_label=out.topk(1,dim=1)
            equals=top_label==labels.view(*top_label.shape)

            accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()

    model.train()   
    print(f"Testing Accuracy: {accuracy/len(dataloaders[2])}")
    print(f"Testing loss: {validation_loss/len(dataloaders[2]):.3f}")
    model.class_to_idx = image_datasets[0].class_to_idx
    checkpoint={'model':model.cpu(),
               'classifier':model.classifier,
               'state_dict':model.state_dict(),
               'optimizer':optimizer.state_dict(),
               'class_to_idx':model.class_to_idx,
               'epochs':5
               }
    torch.save(checkpoint,in_arg.save_dir+'/'+in_arg.arch+'_checkpoint.pth')
    
    
if __name__=='__main__':
    main()
    