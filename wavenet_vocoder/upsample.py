import torch
from torch import nn
from hparams import hparams
from wavenet_vocoder.modules import ConvTranspose2d
import os


class UpSampleConv(nn.Module):
    def __init__(self,
                 path='/home/jinqiangzeng/work/pycharm/P_wavenet_vocoder/checkpoints_teacher/upsample.pth',
                 weight_normalization=True):
        super(UpSampleConv, self).__init__()
        self.path  = path
        self.upsample_conv = nn.ModuleList()
        for s in hparams.upsample_scales:
            freq_axis_padding = (hparams.freq_axis_kernel_size - 1) // 2
            convt = ConvTranspose2d(1, 1, (hparams.freq_axis_kernel_size, s),
                                    padding=(freq_axis_padding, 0),
                                    dilation=1, stride=(1, s),
                                    weight_normalization=weight_normalization)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.ReLU(inplace=True))

        self.load()

    def forward(self, c):
        for f in self.upsample_conv:
            c = f(c)
        return c

    def load(self):
        if self.path and os.path.exists(self.path):
            self.upsample_conv.load_state_dict(torch.load(self.path))
        else:
            raise Exception("can't load state dict, check path, see get_model in train_student.py !")


