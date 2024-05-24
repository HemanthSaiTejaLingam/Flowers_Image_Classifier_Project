import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from torch import nn,optim
from torchvision import datasets,transforms,models
from collections import OrderedDict
from PIL import Image
import argparse
import seaborn as sb

def get_input_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=str,
                       help='path to the folder of image')
    parser.add_argument('checkpoint',type=str,
                       help='path to the saved model')
    parser.add_argument('--category_names',type=str,default='cat_to_name.json',
                       help='categories of names')
    parser.add_argument('--top_k',type=int,default=5,
                       help='top k probabilities of model')
    parser.add_argument('--gpu',action='store_true',
                       help='whether to use gpu or not')
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img=Image.open(image)
    transform=transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    tensor_img=transform(img)
    return tensor_img

def load_checkpoint(filepath):
    checkpoint=torch.load(filepath)
    model=checkpoint['model']
    model.classifier=checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx=checkpoint['class_to_idx']
    optimizer=checkpoint['optimizer']
    epochs=checkpoint['epochs']
    return model,checkpoint


def predict(image_path, model,device,cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    with torch.no_grad():
        tensor_img=process_image(image_path)
        tensor_img=tensor_img.to(device)
        model.to(device)
        tensor_img=tensor_img.unsqueeze(0)
        ps=model.forward(tensor_img)
        out=torch.exp(ps)
        top_prob,top_class=out.topk(topk,dim=1)
        top_prob=top_prob.cpu().numpy().tolist()[0]
        top_class=top_class.cpu().numpy().tolist()[0]
        classes=[]
        flowers=[]
        for cls,idx in model.class_to_idx.items():
            if idx in top_class:
                classes.append(cls)
                flowers.append(cat_to_name[cls])
    model.train()
    #print(top_prob,flowers)
    return top_prob,flowers

    
def main():
    in_args=get_input_args()
    #print(in_args.gpu)
    model,checkpoint_info=load_checkpoint(in_args.checkpoint)
    if in_args.gpu:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device="cpu"
    with open(in_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    p,f=predict(in_args.input,model,device,cat_to_name,in_args.top_k)
    print("probabilities are ",p)
    print("The corresponding flower names are ",f)
 

if __name__=='__main__':
    main()
    
