import audioset.vggish_input as vggish_input
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import torch
import Param
from util import *
from PIL import Image
from scipy import signal
import random
import numpy as np

# os.chdir('')


class DataLoad(Dataset):
    def __init__(self, mode='train', net='vggish', select='malpo', load=False):
        self.net = net
        self.select = select
        self.mode = mode
        self.load = load

        # 区分训练和测试
        if mode == 'train':
            with open('static/train_data.txt', 'r') as f:
                self.feature_folds = eval(f.read())
        elif mode == 'val':
            with open('static/val_data.txt', 'r') as f:
                self.feature_folds = eval(f.read())

        self.audiolist, self.videolist = self.screen_fold()
        print('number:', len(self.feature_folds))

    def __len__(self):
        return len(self.audiolist)

    def __getitem__(self, idx):
        audiopath = self.audiolist[idx]
        videopath = self.videolist[idx]

        if self.net == 'vggsound':
            audio = self.vggsound_process(audiopath)
        else:
            audio = self.vggish_process(audiopath)
        video, video_n = self.video_process(videopath)

        if self.load:
            return audio, video, video_n, self.videolist[idx]
        else:
            return audio, video, video_n

    def screen_fold(self):
        audiolist = []
        videolist = []
        for i in self.feature_folds:
            audiopath = i + '/audio.npy'
            videodir = i + '/video'
            if os.path.exists(audiopath) and os.path.exists(videodir):
                pictures = os.listdir(videodir)
                if len(pictures) >= 32 and (np.load(audiopath, allow_pickle=True)).shape[0] == 128000:
                    videolist.append(videodir)
                    audiolist.append(audiopath)
                else:
                    continue
            else:
                continue
        return audiolist, videolist


    def vggish_process(self, audiopath):
        audio_buffer = np.load(audiopath, allow_pickle=True)
        # print(audio_buffer.shape)
        input_batch = vggish_input.waveform_to_examples(audio_buffer, Param.SAMPLE_RATE)
        input_batch = torch.from_numpy(input_batch).unsqueeze(dim=1)
        audio = input_batch.float()
        return audio

    def vggsound_process(self, audiopath):
        resamples = np.load(audiopath, allow_pickle=True)
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, Param.SAMPLE_RATE, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        return spectrogram

    # visual process
    def read_image(self, small_fold):
        images = open_fold(small_fold)
        images_result = []
        for i in images:
            image = Image.open(i)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)
            images_result.append(image)
        final_x = np.array(images_result)
        return final_x

    def resize(self, buffer):
        resized = []
        for image in buffer:
            im = Image.fromarray(image.astype('uint8')).convert('RGB')
            im = im.resize((Param.FRAME_WIDTH, Param.FRAME_HEIGHT))
            # im.show()
            resized.append(np.array(im))
        return np.array(resized)

    def pretrain(self, buffer, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)):
        trans = [transforms.Normalize(mean=mean, std=std)]
        self.video_transforms = transforms.Compose(trans)
        # buffer = self.crop(buffer)
        # frames * cropedhight * cropedwidth * 3
        # 16 * x * x *3
        buffer = self.resize(buffer)
        buffer = np.array(buffer, dtype='float32')
        buffers = []
        for frame in buffer:
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            buffers.append(np.array(self.video_transforms(frame)))
            # 3*x*x
        buffers = np.array(buffers)
        buffer = torch.tensor(buffers)
        image_final = buffer.permute(1, 0, 2, 3)
        # 3 * 16 * x * x
        return image_final

    def video_process(self, videodir):
        image_buffers = self.read_image(videodir)
        image_buffer = image_buffers[:16]
        if self.select != 'malpo':
            npath = random.choice(self.videolist)
            if npath != videodir:
                image_buffers_n = self.read_image(npath)
                image_buffer_n = image_buffers_n[:16]
            else:
                image_buffer_n = image_buffers[-16:]
        else:
            image_buffer_n = image_buffers[-16:]
        video = self.pretrain(image_buffer)
        video_n = self.pretrain(image_buffer_n)
        return video, video_n

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])
