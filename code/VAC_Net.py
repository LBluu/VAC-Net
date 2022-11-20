import audioset.vggish_postprocess as vggish_postprocess
import torchvision.models as models
from vggish import VGGish
from vggsound import AVENet
import torch.nn.functional as F
import torch.nn as nn
import Param
import torch
import os
import numpy as np

# os.chdir('/')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_parameter_requires_grad(model,feature_extract):
    # 不做更新
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


class MyModel(nn.Module):
    def __init__(self,embed_size= Param.EMBED_SIZE,device =device,net = 'vggish'):
        super(MyModel, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.net = net
        self.audio = self.audio_model()
        self.video = self.video_model()
        self.vggish = nn.Sequential(nn.Linear(Param.DURATION*128,self.embed_size*8),nn.ReLU(),nn.Linear(self.embed_size*8,self.embed_size*4))

        # share weight
        self.share_weight = nn.Sequential(nn.Linear(self.embed_size*4,self.embed_size*2),nn.ReLU(),nn.Linear(self.embed_size*2,self.embed_size))
    
    def forward(self,audio,video,video_n):
        embed_audio = self.audio(audio)
        embed_video = self.video(video)
        embed_video_n = self.video(video_n)

        if self.net == 'vggish':
            embed_audio = self.deal_audio(embed_audio)
            embed_audio = self.vggish(embed_audio)

        audio = self.share_weight(embed_audio)
        video = self.share_weight(embed_video)
        video_n = self.share_weight(embed_video_n)

        return audio,video,video_n

    def deal_audio(self,audio):
        embed_audio = audio.detach().cpu().numpy()
        post_processor = vggish_postprocess.Postprocessor('static/vggish_pca_params.npz')
        embed_audio = post_processor.postprocess(embed_audio)

        resize_audio = []
        tempaudio = 0
        for idx,i in enumerate(embed_audio):
            if type(tempaudio) == int:
                tempaudio = i
            else:
                tempaudio = np.hstack((tempaudio,i))
            if (idx+1) % Param.DURATION ==0:
                resize_audio.append(tempaudio)
                tempaudio = 0
        embed_audio = torch.from_numpy(np.array(resize_audio)).to(self.device)
        embed_audio = embed_audio.float()
        return embed_audio
        

    def audio_model(self):
        if self.net == 'vggsound':
            premodel = AVENet()
            premodel.load_state_dict(torch.load('static/vggsound_netvlad.pth.tar',map_location=self.device)['model_state_dict'])
            set_parameter_requires_grad(premodel,feature_extract=True)
            num_features = premodel.audnet.fc_.in_features
            premodel.audnet.fc_ = nn.Sequential(nn.Linear(num_features,self.embed_size *8),nn.ReLU(),nn.Linear(self.embed_size * 8,self.embed_size *4))
            premodel.audnet.layer4[1].requires_grad_()
            premodel.audnet.avgpool.requires_grad_()
            premodel.audnet.fc_.requires_grad_()
            premodel.to(device)
        else:
            premodel = VGGish()
            premodel.load_state_dict(torch.load('static/pytorch_vggish.pth',map_location=self.device))
            set_parameter_requires_grad(premodel,feature_extract=True)
            premodel.fc.requires_grad_()
            premodel.features[15].requires_grad_()
        return premodel
    
    def video_model(self):
        output_dim = self.embed_size * 4
        v_model = models.video.r2plus1d_18(pretrained=True, progress=True)
        set_parameter_requires_grad(v_model,True)
        num_features = v_model.fc.in_features
        # print(v_model)
        v_model.fc = nn.Sequential(nn.Linear(num_features,self.embed_size *8),nn.ReLU(),nn.Linear(self.embed_size * 8,output_dim))
        v_model.layer4[1].requires_grad_()
        v_model.to(device)
        return v_model