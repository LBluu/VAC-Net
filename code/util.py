import pandas as pd
import os
import cv2
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import Data.PARAMETER as PARAMETER
import plotly.graph_objects as go
import plotly.io as pio
import plotly
import torch

#定义获取函数
def get_df_from_db(db,sql):
    cursor = db.cursor()#使用cursor()方法获取用于执行SQL语句的游标
    cursor.execute(sql)# 执行SQL语句
    """
    使用fetchall函数以元组形式返回所有查询结果并打印出来
    fetchone()返回第一行，fetchmany(n)返回前n行
    游标执行一次后则定位在当前操作行，下一次操作从当前操作行开始
    """
    data = cursor.fetchall()

    #下面为将获取的数据转化为dataframe格式
    columnDes = cursor.description #获取连接对象的描述信息
    columnNames = [columnDes[i][0] for i in range(len(columnDes))] #获取列名
    df = pd.DataFrame([list(i) for i in data],columns=columnNames) #得到的data为二维元组，逐行取出，转化为列表，再转化为df

    """
    使用完成之后需关闭游标和数据库连接，减少资源占用,cursor.close(),db.close()
    db.commit()若对数据库进行了修改，需进行提交之后再关闭
    """
    cursor.close()
    db.close()

    print("cursor.description中的内容：",columnDes)
    return df


def mp3_wav(mp3path,wavpath):
    '''

    :param audiopath: 原audiopath
    :param wavpath: targetpath
    :return:
    '''
    # if os.path.exists(self.audiowav):
    #     return
    # else:

    audiopath = ''
    for i in mp3path:
        if i == ' ' or i == '&':
            audiopath = audiopath+"\\"+i
        else:
            audiopath = audiopath+i

    audiowav = ''
    for i in wavpath:
        if i == ' ' or i == '&':
            audiowav = audiowav+"\\"+i
        else:
            audiowav = audiowav + i

    a = 'ffmpeg -i '+audiopath+' '+audiowav
    os.system(a)


def clip_video(videopath,videodir,count):
    '''
    将要剪切的视频每秒两帧的输入视频的文件夹中
    :param videopath: 要剪切的视频
    :param videodir: 存储视频的文件夹
    :param count: 文件夹中的count
    :return:
    '''
    # print(videopath)
    cap = cv2.VideoCapture(videopath)
    framerate = cap.get(5)

    while(cap.isOpened()):
        frameid = cap.get(1)
        ret,frame = cap.read()
        if (ret!=True):
            break
        if (frameid % math.floor(framerate/2)==0):
            count +=1
            cv2.imwrite(videodir + '/%d.jpg'%count,frame)
            # print(count)
    cap.release()
    return count

def open_fold(fold_path):
    '''
    返回文件夹里的文件的full path
    :param fold_path:
    :return:
    '''
    files = os.listdir(fold_path)
    filepath = []
    for i in files:
        filepath.append(fold_path+'/'+i)
    return filepath


def split_files(foldpath):
    '''
    把这个文件夹里的文件夹里的视频和音频分开
    :param foldpath:BasePath
    :return:各种list
    '''
    flist = open_fold(foldpath)
    videolist = []
    audiolist = []
    danmulist = []
    replylist = []
    for i in flist:
        if os.path.isdir(i):
            files = os.listdir(i)
            if len(files)>=4:
                for j in files:
                    if j[-4:] == '.mp3':
                        audiolist.append(i + '/' + j)
                    elif j[-4:] == '.flv':
                        videolist.append(i + '/' + j)
                danmulist.append(i+'/'+'danmudata.csv')
                replylist.append(i + '/' + 'reply.csv')
    return audiolist,videolist,danmulist,replylist

def read_small_fold(foldpath):
    flist = open_fold(foldpath)
    audiopath,videopath,danmupath,replypath = '','','',''
    for i in flist:
        if i[-4:] == '.mp3':
            audiopath = i
        elif i[-4:] == '.flv':
            videopath = i
        elif i[-5:] == 'a.csv':
            danmupath = i
        elif i[-5:] == 'y.csv':
            replypath = i
    if audiopath!='' and videopath!='' and danmupath!='' and replypath!='':
        return audiopath,videopath,danmupath,replypath
    else:
        return 0,0,0,0

def get_iteminfo():
    foldlist = open_fold(PARAMETER.MAX_FOLD)
    videodir = pd.DataFrame()
    for i in foldlist:
        videodirpath = i + '/videodir.csv'
        if os.path.exists(videodirpath):
            temp = pd.read_csv(videodirpath)
            videodir = pd.concat([temp,videodir],axis=0)
    # idlist = list(videodir['aid'])
    idpathlist = os.listdir(PARAMETER.FEATURE_FOLD)
    finaldir = pd.DataFrame()
    for item in idpathlist:
        finaldir = finaldir.append(videodir.loc[videodir['aid'] == int(item)])
    finaldir.to_csv(PARAMETER.FEATURE_FOLD + '/videodir.csv')
    return 0

def show_plot(iteration,loss,path):
    #绘制损失变化图
    plt.plot(iteration,loss)
    # plt.plot(epoch,val_loss)
    plt.savefig(path)
    plt.close()
    # plt.show()

def show_epoch_plot(ep,loss,valloss):
    plt.plot(ep,loss)
    plt.plot(ep,valloss)
    plt.show()

def get_datalist(item = ''):
    with open(PARAMETER.DATAPATH_TXT,'r') as f:
        datapath  = eval(f.read())
    audio = []
    video = []
    danmu = []
    reply = []
    for small_foldpath in datapath:
        audiopath,videopath,danmupath,replypath = read_small_fold(small_foldpath)
        if audiopath!='' and videopath!='' and danmupath!='' and replypath!='':
            audio.append(audiopath)
            video.append(videopath)
            danmu.append(danmupath)
            reply.append(replypath)
        else:
            continue
    if item == '':
        return audio,video,danmu,reply
    elif item == 'audio':
        return audio
    elif item == 'video':
        return video
    elif item == 'danmu':
        return danmu
    elif item == 'reply':
        return reply
    else:
        return reply

def cosdist(vec1,vec2):
    vec1 = vec1.detach().cpu().numpy()
    vec2 = vec2.detach().cpu().numpy()
    dist1 = np.dot(vec1,vec2.T)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return float(dist1[0])

def draw_hist(x,y,target_file):
    pyplt = plotly.offline.plot
    fig = go.Figure([go.Bar(x=x, y=y)])
    pyplt(fig, filename=target_file)
    return

def process_vggish(audio):
    resize_audio = 0
    for i in audio:
        if type(resize_audio) == int:
            resize_audio = i.data.cpu().numpy()
        else:
            resize_audio = np.concatenate((resize_audio,i.data.cpu().numpy()),axis=0)
    audio = torch.from_numpy(resize_audio)
    return audio

def dist(vec1,vec2):
    vec1 = vec1.detach().cpu().numpy()
    vec2 = vec2.detach().cpu().numpy()
    dist = np.linalg.norm(vec1 - vec2)
    return dist

if  __name__ == '__main__':
    get_iteminfo()