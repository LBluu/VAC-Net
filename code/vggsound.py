import torch
import Param
from torch import nn
import torch.nn.functional as F
from MyNet.VGGSound_models import resnet

class AVENet(nn.Module):

    def __init__(self):
        super(AVENet, self).__init__()
        self.audnet = Resnet()

    def forward(self, audio):
        aud = self.audnet(audio)
        return aud


def Resnet():

    assert Param.MODEL_DEPTH in [10, 18, 34, 50, 101, 152, 200]
    model_depth = Param.MODEL_DEPTH
    n_classes = 309
    pool = "vlad"


    if model_depth == 18:
        model = resnet.resnet18(
            num_classes=n_classes,
            pool=pool)
    elif model_depth == 34:
        model = resnet.resnet34(
            num_classes=n_classes,
            pool=pool)
    return model 

if __name__ == '__main__':
    print(AVENet())
    print(AVENet().audnet.layer4[1])
    print(AVENet().audnet.avgpool)
    print(AVENet().audnet.fc_)