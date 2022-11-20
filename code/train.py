import torch.utils.data as Data
from VAC_Net import MyModel
from load_data import DataLoad
import Param
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from util import *

# os.chdir('')

netmode = 'vggish'
selectmathod = 'malpo'
margin = 10
dirname = '%s_%s_%s'%(netmode,selectmathod,str(margin))
os.mkdir('NetResult/' + dirname)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = MyModel(net=netmode ).to(device)

# 载入数据
train_loader = DataLoad(net=netmode, select=selectmathod)
val_loader = DataLoad(mode='val', net=netmode, select=selectmathod)
train_dataset = Data.DataLoader(train_loader, batch_size=Param.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
val_dataset = Data.DataLoader(val_loader, batch_size=1, pin_memory=True, num_workers=4)

loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

StepLR = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=2*len(train_dataset), epochs=Param.EPOCH)


iteration_number = 0
iter_val = 0
min_corr = 0
val_loss = []


for epoch in tqdm(range(0, Param.EPOCH)):


    for step, data in enumerate(train_dataset):
        audio, video, video_n = data

        if netmode == 'vggish':
            audio = process_vggish(audio)
        else:
            audio = audio.unsqueeze(1).float()

        audio = Variable(audio).cuda()
        video = Variable(video).cuda()
        video_n = Variable(video_n).cuda()

        # 载入训练数据 开始训练
        net.train()
        optimizer.zero_grad()
        audio_embed, video_embed, video_embed_n = net(audio, video, video_n)
        loss = loss_fn(audio_embed, video_embed, video_embed_n)
        loss = loss.requires_grad_()
        loss.backward()
        optimizer.step()
        StepLR.step()


    val_right = 0
    for step, data in enumerate(val_dataset):

        net.eval()
        audio, video, video_n = data
        # print(audio.shape)

        with torch.no_grad():
            if netmode == 'vggish':
                audio = process_vggish(audio)
            else:
                audio = audio.unsqueeze(1).float()

            audio = Variable(audio).cuda()
            video = Variable(video).cuda()
            video_n = Variable(video_n).cuda()

            audio_embed, video_embed, video_embed_n = net(audio, video, video_n)
            val_loss = loss_fn(audio_embed, video_embed, video_embed_n)

        if val_loss < margin:
            val_right = val_right + 1
        val_size = step + 1


    print('Epoch: ' + str(epoch) + '| Current loss : ' + str(loss.item()) + '\n')
    print("Epoch:", epoch, "| Validation Loss:", str(val_loss.item()), '\n')
    if val_right != 0:
        corr = val_right / val_size
    else:
        corr = 0
    print('correct pro', corr)

    if corr > min_corr and corr<0.8:
        print("Validation Loss decreased, saving new best model")
        min_corr = corr
        torch.save(net.state_dict(), 'Net/net_' + dirname + '.pkl')
