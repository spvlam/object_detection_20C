import torch
import torch.nn as nn

""" 
   archtecture_configuration is list of tuple and list
   where each tuple has shape (kernel_size,C_out= number_of_filter, stride, padding)
   list = [(tupe), iteration]
"""

archtecture_configuration = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channel, **kwargs):
        # related to multi inheritance
        #kwargs for padding and stride and more
        super(CNNBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channel,bias=False,**kwargs)
        # batch norm for improve stable trainning and convergence speed
        self.batchNorm = nn.BatchNorm2d(out_channel)
        self.leakyRelu = nn.LeakyReLU(0.1)
    def forward(self,x):
        return self.leakyRelu(self.batchNorm(self.conv(x)))
    
class YOLOv1(nn.Module):
    def __init__(self,inChannel=3, **kwargs) -> None:
        super(YOLOv1,self).__init__()
        self.architecture = archtecture_configuration
        self.inChannel = inChannel
        self.darkNet = self._create_conv_layers(self.architecture)
        self.fcl = self._create_fcl(**kwargs)
    def forward(self,x):
        x = self.darkNet(x)
        return self.fcl(torch.flatten(x,start_dim=1))
    def _create_conv_layers(self,architcture_list):
        layers=[]
        inChannel = self.inChannel
        for architecture in architcture_list:
            if type( architecture ) == str:
                layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
            elif type(architecture) == tuple:
                layers.append(
                    CNNBlock(
                        inChannel,
                        architecture[1],
                        kernel_size = architecture[0],
                        stride=architecture[2],
                        padding=architecture[3])
                )
                inChannel = architecture[1]
            elif type(architecture) == list:
                interation = architecture[2]
                subArchitecture1 = architecture[0]
                subArchitecture2 = architecture[1]
                for inter in range(interation):
                    layers.append(
                        CNNBlock(
                            inChannel,
                            subArchitecture1[1],
                            kernel_size=subArchitecture1[0],
                            padding=subArchitecture1[3],
                            stride=subArchitecture1[2])
                    )
                    inChannel = subArchitecture1[1]
                    layers.append(
                        CNNBlock(
                            inChannel,
                            subArchitecture2[1],
                            kernel_size=subArchitecture2[0],
                            padding=subArchitecture2[3],
                            stride=subArchitecture2[2])
                    )
                    inChannel = subArchitecture2[1]
        return nn.Sequential(*layers)
     
    def _create_fcl(self,numberBox,numberClass,splitFinalSize=7):
        S,B,C = splitFinalSize,numberBox,numberClass
        return nn.Sequential(
            nn.Linear(1024*S*S,496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496,S*S*(C+5*B))
        )
    

# for testing model
def test():
    model = YOLOv1(numberBox=2,numberClass=20)
    x = torch.randn((1,3,448,448))
    print(model(x).shape)
# test()



