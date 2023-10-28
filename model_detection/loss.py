import torch
import torch.nn as nn
from utils_copy import intersection_over_union

class YoloLoss(nn.Module):
    """ 
     calculate the loss function of yolov1 follow by document
    """
    def __init__(self,S=7,B=2,C=20):
        super(YoloLoss,self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S=S
        self.B=B
        self.C=C
        self.lambda_noobj =0.5
        self.lambda_coord = 5
    def forword(self,predictions,target):
        """_summary_

        Args:
            predictions (tensor): (batchSize,S*S*30)
            30-D = (C0,C1,C2,..,C20,Pr1,x1,y1,w1,h1,Pr2,x2,y2,w2,h2)
            target (tensor):(batch,S,S,25)
            25-D = (C0,C1,C2,...,C20,Pr1,x1,y1,w1,h1)
        """
        predictions = predictions.reshape(-1,self.S,self.S,self.C+5*self.B)
        iou_b1= intersection_over_union(predictions[...,21:25],target[...,21:25])
        iou_b2= intersection_over_union(predictions[...,26:30],target[...,21:25])
        ious = torch.cat((iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)),dim=0)
        iouMax, bestBox = torch.max(ious,dim=0)
        existedBox = target[...,20].unsqueeze(3)
        # coordinated loss
        # predicted_box 

        
