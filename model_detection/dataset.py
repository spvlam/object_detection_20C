import torch
import os
import pandas as pd
from PIL import Image
""" 
   method start with __ and end with __ is call magical method or dunder method = double underscore method
"""
class VOCDataset(torch.utils.data.Dataset):
    def __init__(
      self,csv_file,img_dir,label_dir,S=7,B=2,C=20,transform=None      
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
    def __len__(self):
        return len(self.annotations)    
    def __getitem__(self,index):
    # index is the index of imagine
    # return : imagine and label_tensor (s,s,25)
        label_path = os.path.join(self.label_dir,self.annotations.iloc[index,1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_labels,x,y,w,h = [
                float(x) if '.' in x else int(x)
                for x in label.replace('\n','').split()
                ]
            boxes.append([class_labels,x,y,w,h])
        img_path = os.path.join(self.img_dir,self.annotations.iloc[index,0])
        img = Image.open(img_path)
##############################################
## I DO NOT KNOW HOW TRANSFORM HAS BEEN USE ##
##############################################
        if self.transform:
            img,boxes = self.transform(img,boxes)
        label_tenfor = torch.zeros(self.S,self.S,25)
        for box in boxes:
            class_labels,x,y,w,h = box
            class_labels = int(class_labels)
            # i , j for row and collumn respectively
            i,j = int(self.S*y), int(self.S*x)
            cell_x, cell_y = self.S*x - j , self.S*y-i
            width_px,height_px = (
                self.S*w,self.S*h
            )
            if label_tenfor[i,j,20]==0:
                label_tenfor[i,j,20]=1
                label_tenfor[i,j,21:25] = torch.tensor([cell_x,cell_y,width_px,height_px])
                label_tenfor[i,j,class_labels]=1
        return img,label_tenfor




