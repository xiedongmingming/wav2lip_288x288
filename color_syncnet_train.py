from os.path import dirname, join, basename, isfile

from tqdm import tqdm

from models import SyncNet_color as SyncNet

import audio

import torch

from torch import nn
from torch import optim

import torch.backends.cudnn as cudnn

from torch.utils import data as data_utils

import numpy as np

from glob import glob

import os, random, cv2, argparse

from hparams import hparams, get_image_list

parser = argparse.ArgumentParser(description='code to train the expert lip-sync discriminator')

parser.add_argument(
    "--data_root",
    help="root folder of the preprocessed lrs2 dataset",
    required=True
)
parser.add_argument(
    '--checkpoint_dir',
    help='save checkpoints to this directory',
    required=True,
    type=str
)
parser.add_argument(
    '--checkpoint_path',
    help='resumed from this checkpoint',
    default=None,
    type=str
)

args = parser.parse_args()

########################################################
# 全局变量
global_steps = 0  # 以EPOCH为周期
global_epoch = 0  # 一个EPOCH表示遍历一遍训练数据集

use_cuda = torch.cuda.is_available()

# print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16


########################################################
class Dataset(object):
    #
    def __init__(self, split):

        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):

        return int(basename(frame).split('.')[0])  # 图片索引编号

    def get_window(self, start_frame):

        start_id = self.get_frame_id(start_frame)

        vidname = dirname(start_frame)

        window_fnames = []

        for frame_id in range(start_id, start_id + syncnet_T):  # 随后的200MS时长内的图片

            frame = join(vidname, '{}.jpg'.format(frame_id))

            if not isfile(frame):
                #
                return None

            window_fnames.append(frame)

        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        #
        # num_frames = (T x hop_size * fps) / sample_rate
        #
        start_frame_num = self.get_frame_id(start_frame)

        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def __len__(self):
        #
        return len(self.all_videos)

    def __getitem__(self, idx):

        while 1:

            idx = random.randint(0, len(self.all_videos) - 1)

            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))

            if len(img_names) <= 3 * syncnet_T:
                #
                # print('视频帧总是不足：{}'.format(vidname))
                #
                continue

            right_img_name = random.choice(img_names)  # 正样本
            wrong_img_name = random.choice(img_names)  # 负样本

            while wrong_img_name == right_img_name:
                #
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):  # 构造正标签

                y = torch.ones(1).float()

                chosen = right_img_name

            else:  # 构造负标签

                y = torch.zeros(1).float()

                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)

            if window_fnames is None:
                #
                # print('视频窗口为空：{}'.format(chosen))
                #
                continue

            window = []

            all_read = True

            for fname in window_fnames:
                #
                img = cv2.imread(fname)

                if img is None:
                    #
                    all_read = False

                    break

                try:

                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))  # 调整大小

                except Exception as e:

                    print('视频帧调整尺寸异常：{}->{}'.format(fname, e))

                    all_read = False

                    break

                window.append(img)

            if not all_read:
                #
                print('视频窗口帧读取失败：{}'.format(chosen))

                continue

            try:

                wavpath = join(vidname, "audio.wav")

                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T

            except Exception as e:

                print('视频片段的音频数据读取异常：{}'.format(e))

                continue

            mel = self.crop_audio_window(orig_mel.copy(), right_img_name)  # 正样本对应的音频数据：{ndarray: {16, 80}}

            if mel.shape[0] != syncnet_mel_step_size:
                #
                continue
            #
            # H x W x 3 * T
            #
            x = np.concatenate(window, axis=2) / 255.  # {ndarray: (288, 288, 15)}

            x = x.transpose(2, 0, 1)  # {ndarray: (15, 288, 288)}
            x = x[:, x.shape[1] // 2:]  # {ndarray: (15, 288//2, 288)}--将第一个数字除以第二个数字并将结果向下舍入为最接近的整数

            x = torch.FloatTensor(x)  # {Tensor: (15, 288//2, 288)}

            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            # x: {Tensor: (15, 144, 288)} --> 正/负样本的输入(200MS内5张图片数据合并--只保留下半部分)：[N, C, H, W]
            # y: {Tensor: (1,)}           --> 正负样本对应的标签
            # mel: {Tensor: (1, 80, 16)}  --> 正样本时200MS内的音频数：80--属性数量；16--样本数

            return x, mel, y


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    #
    # a--audio
    # v--faces
    # y--labels
    #
    d = nn.functional.cosine_similarity(a, v)  # 距离越近COSINE越接近1（否则越接近0）

    loss = logloss(d.unsqueeze(1), y)

    return loss


def train(
        device,
        syncnet_model,
        train_data_loader,
        test_data_loader,
        syncnet_optimizer,
        checkpoint_dir=None,
        checkpoint_interval=None,
        nepochs=None
):
    #
    global global_steps, global_epoch

    resumed_steps = global_steps

    while global_epoch < nepochs:

        running_loss = 0.

        prog_bar = tqdm(enumerate(train_data_loader))

        for step, (x, mel, y) in prog_bar:
            #
            # x: {Tensor: (15, 144, 288)} --> 正/负样本的输入(200MS内5张图片数据合并--只保留下半部分)：[N, C, H, W]
            # y: {Tensor: (1,)}           --> 正负样本对应的标签
            # mel: {Tensor: (1, 80, 16)}  --> 正样本时200MS内的音频数：80--属性数量；16--样本数
            #
            syncnet_model.train()

            syncnet_optimizer.zero_grad()

            x = x.to(device)  # transform data to cuda device

            mel = mel.to(device)

            a, v = syncnet_model(mel, x)  # 200MS音频+5张下半部分脸部数据->AUDIO+FACES

            y = y.to(device)

            loss = cosine_loss(a, v, y)

            loss.backward()

            syncnet_optimizer.step()

            global_steps += 1

            cur_session_steps = global_steps - resumed_steps

            running_loss += loss.item()  # LOSS和值

            if global_steps == 1 or global_steps % checkpoint_interval == 0:
                #
                save_checkpoint(
                    syncnet_model,
                    syncnet_optimizer,
                    global_steps,
                    checkpoint_dir,
                    global_epoch
                )

            if global_steps % hparams.syncnet_eval_interval == 0:
                #
                with torch.no_grad():
                    #
                    eval_model(
                        test_data_loader,
                        global_steps,
                        device,
                        syncnet_model,
                        checkpoint_dir
                    )

            prog_bar.set_description('loss: {}'.format(running_loss / (step + 1)))

        global_epoch += 1


def eval_model(test_data_loader, global_steps, device, model, checkpoint_dir):
    #
    eval_steps = 1400

    print('evaluating for {} steps'.format(eval_steps))

    losses = []

    while 1:

        for step, (x, mel, y) in enumerate(test_data_loader):
            #
            # x: {Tensor: (15, 144, 288)} --> 正/负样本的输入(200MS内5张图片数据合并--只保留下半部分)：[N, C, H, W]
            # y: {Tensor: (1,)}           --> 正负样本对应的标签
            # mel: {Tensor: (1, 80, 16)}  --> 正样本时200MS内的音频数：80--属性数量；16--样本数
            #
            model.eval()

            x = x.to(device)  # transform data to cuda device

            mel = mel.to(device)

            a, v = model(mel, x)

            y = y.to(device)

            loss = cosine_loss(a, v, y)

            losses.append(loss.item())

            if step > eval_steps:
                #
                break

        averaged_loss = sum(losses) / len(losses)

        print(averaged_loss)  # 平均损失值

        return


def save_checkpoint(model, optimizer, steps, checkpoint_dir, epoch):
    #
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_steps)
    )

    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None

    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer_state,
            "global_steps": steps,
            "global_epoch": epoch,
        },
        checkpoint_path
    )

    print("saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    #
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    #
    global global_steps
    global global_epoch

    print("load checkpoint from: {}".format(path))

    checkpoint = _load(path)

    model.load_state_dict(checkpoint["state_dict"])

    if not reset_optimizer:  # 需要加载优化器

        optimizer_state = checkpoint["optimizer"]

        if optimizer_state is not None:
            #
            print("load optimizer state from {}".format(path))

            optimizer.load_state_dict(checkpoint["optimizer"])

    global_steps = checkpoint["global_steps"]  # 这俩是定制参数
    global_epoch = checkpoint["global_epoch"]

    return model


if __name__ == "__main__":
    #
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir):
        #
        os.mkdir(checkpoint_dir)

    # dataset and dataloader setup
    train_dataset = Dataset('train')
    tests_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(  # 会将所有数据分批次遍历一遍
        train_dataset,
        batch_size=hparams.syncnet_batch_size,
        shuffle=True,
        num_workers=hparams.num_workers
    )

    tests_data_loader = data_utils.DataLoader(
        tests_dataset,
        batch_size=hparams.syncnet_batch_size,
        # shuffle=True,
        num_workers=8
    )

    device = torch.device("cuda" if use_cuda else "cpu")

    ###################################################################
    # 模型
    syncnet_model = SyncNet().to(device)

    print('total trainable params: syncnet {}'.format(
        sum(p.numel() for p in syncnet_model.parameters() if p.requires_grad)
    ))

    syncnet_optimizer = optim.Adam(
        [p for p in syncnet_model.parameters() if p.requires_grad],
        lr=hparams.syncnet_lr
    )

    if checkpoint_path is not None:
        #
        load_checkpoint(checkpoint_path, syncnet_model, syncnet_optimizer, reset_optimizer=False)

    train(
        device,
        syncnet_model,
        train_data_loader,
        tests_data_loader,
        syncnet_optimizer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=hparams.syncnet_checkpoint_interval,
        nepochs=hparams.nepochs
    )
