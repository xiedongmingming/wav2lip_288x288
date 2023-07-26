import torch

from torch import nn
from torch.nn import functional as F


class Conv2d(nn.Module):

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        #
        super().__init__(*args, **kwargs)

        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),  # 二维卷积
            nn.BatchNorm2d(cout)  # 二维批量归一化层：批量归一化是一种用于提高深度学习模型性能和稳定性的技术
        )

        self.act = nn.ReLU()  # TODO nn.PReLU()：原WAV2LIP模型就是使用的RELU()

        self.residual = residual

    def forward(self, x):
        #
        out = self.conv_block(x)  # {Tensor: (32, 15, 144, 288)} --> [N, C, H/2, W]

        if self.residual: # {Tensor: (32, COUT, 144, 288)} --> [N, C, H/2, W]
            #
            out += x

        return self.act(out)


class nonorm_Conv2d(nn.Module):  # 不带归一化的卷积

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        #
        super().__init__(*args, **kwargs)

        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
        )

        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)  # TODO 原WAV2LIP模型有：inplace=True

    def forward(self, x):
        #
        out = self.conv_block(x)

        return self.act(out)


class Conv2dTranspose(nn.Module):

    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs): # output_padding
        #
        super().__init__(*args, **kwargs)

        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout)
        )

        self.act = nn.ReLU()  # TODO nn.PReLU()：原WAV2LIP模型就是使用的RELU()

    def forward(self, x):
        #
        out = self.conv_block(x)

        return self.act(out)
