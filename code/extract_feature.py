import cv2
from util import *
import Param
import math
import random
import librosa
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
from tqdm import tqdm
'''
1. 筛选弹幕数较少的
2. 将选取的视频样本合成视频方便人工筛选
3. 输出：
    Feature 文件夹有：audio.npy、video分帧文件夹
    Sample 文件夹有：取名和Feature一样的文件夹 内有正负样本的视频
'''


PERCENT = 75


class SelectData:
    def __init__(self):
        with open(Param.DATA_FOLDS,'r') as f:
            self.folds = eval(f.read())

    def one_video(self,smallfold):
        self.audiopath,self.videopath,self.danmupath,self.replypath = read_small_fold(smallfold)

        self.name = smallfold.split('/')[-1]
        offsetlist = self.find_offset()
        # 获得初始时间段节点
        for num in range(0,len(offsetlist)):
            flag = self.form_sample(offsetlist,num)
            if flag ==0:
                return 0
            # 形成sample文件夹
            flag = self.form_feature(offsetlist,smallfold,num)
            # 形成feature文件夹
            if flag ==0:
                return 0
        return 1


    def find_offset(self):
        '''
        数据的文件夹
        :param smallfold:
        :return: offset 的列表
        '''
        offset = []
        try:
            danmu = pd.read_csv(self.danmupath,engine='python')
        except:
            return 0

        number = {}
        # 分段计算弹幕数量
        for row in danmu.itertuples():
            second = getattr(row,'second')
            if str(second) == 'nan':
                continue
            # 按照8s分割
            period = int(second // Param.DURATION)
            if period not in list(number.keys()):
                number[period] = 1
            else:
                number[period] +=1

        y=[]
        for key in number.keys():
            y.append(number[key])
        stand = np.percentile(np.array(y), PERCENT)

        if len(list(number.keys())) > 4:
            for t in number.keys():
                if number[t] > stand and t < list(number.keys())[-2] and t>list(number.keys())[1]:
                        offset.append(t*Param.DURATION)
        return offset

    def form_sample(self,offset,num):
        '''
        输入开始时间 small文件夹 offset的index
        形成在Sample中的文件夹
        :return:
        '''
        # self.audiopath,self.videopath,self.danmupath,self.replypath = read_small_fold(smallfold)
        offset = offset[num]
        # 在Sample中建立文件夹
        dirpath = Param.SAMPLE_DIR + '/' + self.name + str(num)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        target_file = dirpath+'/video.mp4'
        target_file_n = dirpath + '/video_n.mp4'
        self.clipedvideo = target_file
        self.clipedvideo_n = target_file_n
        if os.path.exists(target_file) and os.path.exists(target_file_n):
            return


        # 剪切视频
        video_target_file = dirpath+'/temp_video.mp4'
        video_n_target_file = dirpath+'/temp_video_n.mp4'
        try:
            source_video = VideoFileClip(self.videopath)
        except:
            return 0
        video = source_video.subclip(offset, offset+Param.DURATION) # 执行剪切操作
        video.write_videofile(video_target_file)
        video = source_video.subclip(offset+Param.MALPOS, offset+Param.MALPOS+Param.DURATION)  # 执行剪切操作
        video.write_videofile(video_n_target_file)


        # 剪切音频
        try:
            audio = AudioSegment.from_file(self.audiopath, "mp4")
        except:
            return 0
        audio = audio[offset * 1000: (offset+Param.DURATION) * 1000]
        audio_target_file = dirpath + '/audio.mp4'
        audio_format = audio_target_file[audio_target_file.rindex(".") + 1:]
        audio.export(audio_target_file, format=audio_format)

        # 合并
        # 注：需要先指定音频再指定视频，否则可能出现无声音的情况

        command = "ffmpeg -y -i {0} -i {1} -vcodec copy -acodec copy {2}".format(audio_target_file, video_target_file, target_file)
        os.system(command)
        command = "ffmpeg -y -i {0} -i {1} -vcodec copy -acodec copy {2}".format(audio_target_file, video_n_target_file, target_file_n)
        os.system(command)

        os.remove(video_target_file)
        os.remove(audio_target_file)
        os.remove(video_n_target_file)

        return

    def form_feature(self,offset,smallfold,num):
        '''
        :param offset: 开始时间
        :param smallfold:
        :return:
        '''
        offset = offset[num]

        wavpath = smallfold+'/audio.wav'
        if not os.path.exists(wavpath):
            self.mp3_wav(self.audiopath,wavpath)
        # wav文件保存在smallfold 里
        if not os.path.exists(wavpath):
            return 0

        # 建立Feature下的文件夹
        dirpath = Param.FEATURE_DIR + '/' + self.name + str(num)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)


        audiopath = dirpath + '/audio.npy'
        if not os.path.exists(audiopath):
            # 提取音频
            data, sr = librosa.load(wavpath, sr=Param.SAMPLE_RATE, dtype='float32', offset=offset, duration=Param.DURATION)
            np.save(audiopath,data)

        videodir = dirpath + '/video'
        if not os.path.exists(videodir):
            os.mkdir(videodir)
        else:
            if len(os.listdir(videodir)) > 32:
                return 1

        # 剪切视频
        count = 0
        count = self.clip_video(self.clipedvideo,videodir,count)
        count = self.clip_video(self.clipedvideo_n,videodir,count)
        if count < 16:
            # print(count)
            return 0
        else:
            return 1

    def clip_video(self,videopath,videodir,count):
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


    def mp3_wav(self,mp3path,wavpath):
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

