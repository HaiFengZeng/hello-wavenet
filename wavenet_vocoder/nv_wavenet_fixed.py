# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import math
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from .modules import Embedding

from .modules import Conv1d1x1, ResidualConv1dGLU, ConvTranspose2d
from .mixture import sample_from_discretized_mix_logistic
from hparams import hparams
from .modules import _conv1x1_forward


def _expand_global_features(B, T, g, bct=True):
    """Expand global conditioning features to all time steps

    Args:
        B (int): Batch size.
        T (int): Time length.
        g (Variable): Global features, (B x C) or (B x C x 1).
        bct (bool) : returns (B x C x T) if True, otherwise (B x T x C)

    Returns:
        Variable: B x C x T or B x T x C or None
    """
    if g is None:
        return None
    g = g.unsqueeze(-1) if g.dim() == 2 else g
    if bct:
        g_bct = g.expand(B, -1, T)
        return g_bct.contiguous()
    else:
        g_btc = g.expand(B, -1, T).transpose(1, 2)
        return g_btc.contiguous()


def receptive_field_size(total_layers, num_cycles, kernel_size,
                         dilation=lambda x: 2 ** x):
    """Compute receptive field size

    Args:
        total_layers (int): total layers
        num_cycles (int): cycles
        kernel_size (int): kernel size
        dilation (lambda): lambda to compute dilation factor. ``lambda x : 1``
          to disable dilated convolution.

    Returns:
        int: receptive field size in sample

    """
    assert total_layers % num_cycles == 0
    layers_per_cycle = total_layers // num_cycles
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) + 1


class Conv2X1(nn.Module):
    def __init__(self, feature_input, feature_output, conv_type='casual'):
        super(Conv2X1, self).__init__()
        self.conv_type = conv_type
        bias = True if conv_type == 'casual' else False
        self.w_pre = nn.Linear(feature_input, feature_output, bias=bias)
        self.w_cur = nn.Linear(feature_input, feature_output, bias=bias)

    def forward(self, _pre=None, _cur=None):
        if self.conv_type == 'casual':
            B, _, T = _pre.size()
            self.embedding_prev = torch.cuda.FloatTensor(B, hparams.residual_channels, T).fill_(0)
            self.embedding_curr = torch.cuda.FloatTensor(B, hparams.residual_channels, T).fill_(0)
            # _pre [B,A,T]
            for i in range(T):
                self.embedding_prev[:, :, i] = self.w_pre(_pre[:, :, i])  # BXR
                self.embedding_curr[:, :, i] = self.w_pre(_pre[:, :, i])  # BXR
            ret = self.embedding_prev + self.embedding_curr
            return ret  # BXRXT
        elif self.conv_type == 'dilated':
            _pre = _pre.transpose(1, 2)
            _cur = _cur.transpose(1, 2)
            a_pre = self.w_pre(_pre)
            a_cur = self.w_cur(_cur)
            a_pre = a_pre.transpose(1, 2)
            a_cur = a_cur.transpose(1, 2)
            return a_pre, a_cur  # BX2RXT


def get_pre_cur(x, dilated=0, conv_type='casual'):
    # x [B,A,T] A input one-hot channel
    a_pre, a_cur = None, None
    if conv_type == 'casual':
        x_pad = F.pad(x, (1, 0))
        a_pre, a_cur = x_pad[:, :, :-1], x_pad[:, :, 1:]
    elif conv_type == 'dilated':
        x_pad = F.pad(x, (dilated, 0))
        a_pre, a_cur = x_pad[:, :, :-dilated], x_pad[:, :, dilated:]
    return a_pre, a_cur


class ResidualBlock(nn.Module):
    def __init__(self, dilation):
        super(ResidualBlock, self).__init__()
        self.bias_h = nn.Parameter(torch.cuda.FloatTensor(hparams.gate_channels))
        self.residual = nn.Linear(hparams.gate_channels // 2, hparams.residual_channels, bias=True)
        self.dilation = dilation

    def forward(self, h_pre, h_cur, condition):
        # [B,2*R,T]?
        if condition is not None:
            a = h_pre + h_cur + condition + self.bias_h
            # a = h_pre + h_cur + self.bias_h + condition
        else:
            a = h_pre + h_cur + self.bias_h
        a_0, a_1 = a[:, :hparams.residual_channels, :], a[:, hparams.residual_channels:, :]
        h = torch.tanh(a_0) * torch.sigmoid(a_1)

        h = h.transpose(1, 2)
        x = self.residual(h)
        x = x.transpose(1, 2)
        h = h.transpose(1, 2)
        return x, h

    def get_weights(self):
        return self.residual.weight.data, self.residual.bias.data


class NvWaveNet(nn.Module):
    """The WaveNet model that supports local and global conditioning.

    Args:
        out_channels (int): Output channels. If input_type is mu-law quantized
          one-hot vecror. this must equal to the quantize channels. Other wise
          num_mixtures x 3 (pi, mu, log_scale).
        layers (int): Number of total layers
        stacks (int): Number of dilation cycles
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        skip_out_channels (int): Skip connection channels.
        kernel_size (int): Kernel size of convolution layers.
        dropout (float): Dropout probability.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        n_speakers (int): Number of speakers. Used only if global conditioning
          is enabled.
        weight_normalization (bool): If True, DeepVoice3-style weight
          normalization is applied.
        upsample_conditional_features (bool): Whether upsampling local
          conditioning features by transposed convolution layers or not.
        upsample_scales (list): List of upsample scale.
          ``np.prod(upsample_scales)`` must equal to hop size. Used only if
          upsample_conditional_features is enabled.
        freq_axis_kernel_size (int): Freq-axis kernel_size for transposed
          convolution layers for upsampling. If you only care about time-axis
          upsampling, set this to 1.
        scalar_input (Bool): If True, scalar input ([-1, 1]) is expected, otherwise
          quantized one-hot vector is expected.
        use_speaker_embedding (Bool): Use speaker embedding or Not. Set to False
          if you want to disable embedding layer and use external features
          directly.
    """

    def __init__(self, out_channels=256, layers=20, stacks=2,
                 residual_channels=64,
                 gate_channels=64 * 2,
                 skip_out_channels=256,
                 kernel_size=2, dropout=1 - 0.98,
                 cin_channels=-1,
                 gin_channels=-1, n_speakers=None,
                 weight_normalization=True,
                 upsample_conditional_features=False,
                 upsample_scales=None,
                 freq_axis_kernel_size=3,
                 scalar_input=False,
                 use_speaker_embedding=True,
                 ):
        super(NvWaveNet, self).__init__()
        self.scalar_input = scalar_input
        self.out_channels = out_channels
        self.cin_channels = cin_channels
        self.gin_channels = gin_channels
        self.max_dilation = 2 ** (layers / stacks)
        self.layers = layers
        self.stacks = stacks
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        # casual layer
        self.first_conv = Conv2X1(out_channels, residual_channels)
        # layers
        self.conv_layers = nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv_nv = Conv2X1(residual_channels, gate_channels, conv_type='dilated')
            self.add_module('dilated_{}'.format(layer), conv_nv)  # dilated layer
            residual = ResidualBlock(dilation)
            self.add_module('residual_{}'.format(layer), residual)  # residual layer
            skip_out = nn.Linear(residual_channels, skip_out_channels, bias=True)
            self.add_module('skip_out_{}'.format(layer), skip_out)  # skip_out layer
            if cin_channels > 0:
                conv1x1c = Conv1d1x1(cin_channels,
                                     gate_channels,
                                     bias=True,
                                     weight_normalization=weight_normalization)
                self.add_module('c_{}'.format(layer), conv1x1c)
            if gin_channels > 0:
                conv1x1g = Conv1d1x1(gin_channels,
                                     gate_channels,
                                     bias=True,
                                     weight_normalization=weight_normalization)
                self.add_module('g_{}'.format(layer), conv1x1g)
        # last output
        self.relu_out = nn.Linear(skip_out_channels, skip_out_channels)
        self.softmax_out = nn.Linear(skip_out_channels, out_channels)

        if gin_channels > 0:
            assert n_speakers is not None
            self.embed_speakers = Embedding(n_speakers, gin_channels, padding_idx=None, std=0.1)
        else:
            self.embed_speakers = None

        # Upsample
        if upsample_conditional_features:
            self.upsample_conv = nn.ModuleList()
            for s in upsample_scales:
                freq_axis_padding = (freq_axis_kernel_size - 1) // 2
                convt = ConvTranspose2d(1, 1, (freq_axis_kernel_size, s),
                                        padding=(freq_axis_padding, 0),
                                        dilation=1, stride=(1, s),
                                        weight_normalization=weight_normalization)
                self.upsample_conv.append(convt)
                # assuming we use [0, 1] scaled features
                # this should avoid non-negative upsampling output
                self.upsample_conv.append(nn.ReLU(inplace=True))
        else:
            self.upsample_conv = None

        self.receptive_field = receptive_field_size(layers, stacks, kernel_size)

    def has_speaker_embedding(self):
        return self.embed_speakers is not None

    def local_conditioning_enabled(self):
        return self.cin_channels > 0

    def forward(self, x, c=None, g=None, softmax=True):
        """Forward step

        Args:
            x (Variable): One-hot encoded audio signal, shape (B x C x T)
            c (Variable): Local conditioning features,
              shape (B x cin_channels x T)
            g (Variable): Global conditioning features,
              shape (B x gin_channels x 1) or speaker Ids of shape (B x 1).
              Note that ``self.use_speaker_embedding`` must be False when you
              want to disable embedding layer and use external features
              directly (e.g., one-hot vector).
              Also type of input tensor must be FloatTensor, not LongTensor
              in case of ``self.use_speaker_embedding`` equals False.
            softmax (bool): Whether applies softmax or not.

        Returns:
            Variable: output, shape B x out_channels x T
        """
        B, _, T = x.size()

        if g is not None:
            if self.embed_speakers is not None:
                # (B x 1) -> (B x 1 x gin_channels)
                g = self.embed_speakers(g.view(B, -1))
                # (B x gin_channels x 1)
                g = g.transpose(1, 2)
                assert g.dim() == 3
        # Expand global conditioning features to all time steps
        g_bct = _expand_global_features(B, T, g, bct=True)

        if self.upsample_conv is not None:
            # B x 1 x C x T
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
            c = c.squeeze(1)

        # Feed data to network
        pre, cur = get_pre_cur(x)
        x = self.first_conv(pre, cur)
        skips = None
        index = 0
        layers_per_stack = self.layers / self.stacks
        for layer in range(self.layers):
            dilation = 2 ** int(layer % layers_per_stack)
            condition = None
            if c is not None:
                c_f = self._modules['c_{}'.format(layer)]
                _c = _conv1x1_forward(c_f, c, False)
                condition += _c
            if g is not None:
                g_f = self._modules['g_{}'.format(layer)]
                _g = _conv1x1_forward(g_f, g_bct, False)
                condition += _g

            dilated_f = self._modules['dilated_{}'.format(layer)]
            pre, cur = get_pre_cur(x, dilated=dilation, conv_type='dilated')
            a_pre, a_cur = dilated_f(pre, cur)
            residual_f = self._modules['residual_{}'.format(layer)]
            x, h = residual_f(a_pre, a_cur, condition)
            skip_out_f = self._modules['skip_out_{}'.format(layer)]
            h = h.transpose(1, 2)
            skip = skip_out_f(h)
            if index == 0:
                skips = skip
            else:
                skips += skip
            index += 1

        zs = F.relu(skips)
        za = F.relu(self.relu_out(zs))
        za = self.softmax_out(za)
        za = za.transpose(1, 2)
        x = F.softmax(za, dim=2) if softmax else x
        return x

    def incremental_forward(self, initial_input=None, c=None, g=None,
                            T=100, test_inputs=None,
                            tqdm=lambda x: x, softmax=True, quantize=True,
                            log_scale_min=-7.0):
        """Incremental forward step

        Due to linearized convolutions, inputs of shape (B x C x T) are reshaped
        to (B x T x C) internally and fed to the network for each time step.
        Input of each time step will be of shape (B x 1 x C).

        Args:
            initial_input (Variable): Initial decoder input, (B x C x 1)
            c (Variable): Local conditioning features, shape (B x C' x T)
            g (Variable): Global conditioning features, shape (B x C'' or B x C''x 1)
            T (int): Number of time steps to generate.
            test_inputs (Variable): Teacher forcing inputs (for debugging)
            tqdm (lamda) : tqdm
            softmax (bool) : Whether applies softmax or not
            quantize (bool): Whether quantize softmax output before feeding the
              network output to input for the next time step. TODO: rename
            log_scale_min (float):  Log scale minimum value.

        Returns:
            Variable: Generated one-hot encoded samples. B x C x T　
              or scaler vector B x 1 x T
        """
        self.clear_buffer()
        B = 1

        # Note: shape should be **(B x T x C)**, not (B x C x T) opposed to
        # batch forward due to linealized convolution
        if test_inputs is not None:
            if self.scalar_input:
                if test_inputs.size(1) == 1:
                    test_inputs = test_inputs.transpose(1, 2).contiguous()
            else:
                if test_inputs.size(1) == self.out_channels:
                    test_inputs = test_inputs.transpose(1, 2).contiguous()

            B = test_inputs.size(0)
            if T is None:
                T = test_inputs.size(1)
            else:
                T = max(T, test_inputs.size(1))
        # cast to int in case of numpy.int64...
        T = int(T)

        # Global conditioning
        if g is not None:
            if self.embed_speakers is not None:
                g = self.embed_speakers(g.view(B, -1))
                # (B x gin_channels, 1)
                g = g.transpose(1, 2)
                assert g.dim() == 3
        g_btc = _expand_global_features(B, T, g, bct=False)

        # Local conditioning
        if self.upsample_conv is not None:
            assert c is not None
            # B x 1 x C x T
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
            c = c.squeeze(1)
            assert c.size(-1) == T
        if c is not None and c.size(-1) == T:
            c = c.transpose(1, 2).contiguous()

        outputs = []
        if initial_input is None:
            if self.scalar_input:
                initial_input = Variable(torch.zeros(B, 1, 1))
            else:
                initial_input = Variable(torch.zeros(B, 1, self.out_channels))
                initial_input[:, :, 127] = 1  # TODO: is this ok?
            # https://github.com/pytorch/pytorch/issues/584#issuecomment-275169567
            if next(self.parameters()).is_cuda:
                initial_input = initial_input.cuda()
        else:
            if initial_input.size(1) == self.out_channels:
                initial_input = initial_input.transpose(1, 2).contiguous()

        current_input = initial_input

        for t in tqdm(range(T)):
            if test_inputs is not None and t < test_inputs.size(1):
                current_input = test_inputs[:, t, :].unsqueeze(1)
            else:
                if t > 0:
                    current_input = outputs[-1]
                    current_input = Variable(current_input)

            # Conditioning features for single time step
            ct = None if c is None else c[:, t, :].unsqueeze(1)
            gt = None if g is None else g_btc[:, t, :].unsqueeze(1)

            x = current_input
            x = self.first_conv.incremental_forward(x)
            skips = None
            for f in self.conv_layers:
                x, h = f.incremental_forward(x, ct, gt)
                skips = h if skips is None else (skips + h) * math.sqrt(0.5)
            x = skips
            for f in self.last_conv_layers:
                try:
                    x = f.incremental_forward(x)
                except AttributeError:
                    x = f(x)

            # Generate next input by sampling
            if self.scalar_input:
                x = sample_from_discretized_mix_logistic(
                    x.view(B, -1, 1), log_scale_min=log_scale_min)
            else:
                x = F.softmax(x.view(B, -1), dim=1) if softmax else x.view(B, -1)
                if quantize:
                    sample = np.random.choice(
                        np.arange(self.out_channels), p=x.view(-1).data.cpu().numpy())
                    x.zero_()
                    x[:, sample] = 1.0
            outputs += [x.data]
        # T x B x C
        outputs = torch.stack(outputs)
        # B x C x T
        outputs = outputs.transpose(0, 1).transpose(1, 2).contiguous()

        self.clear_buffer()
        return outputs

    def clear_buffer(self):
        self.first_conv.clear_buffer()
        for f in self.conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)

    def get_state_dict(self):
        def get_parameters_list(model, name):
            weights_list, bias_list = [], []
            for layer in range(model.layers):
                weight = model._modules[name + '_{}'.format(layer)].weight.data
                bias = model._modules[name + '_{}'.format(layer)].bias.data
                if name == 'residual':
                    weight = model._modules[name + '_{}'.format(layer)].residual.weight.data
                    bias = model._modules[name + '_{}'.format(layer)].residual.bias.data
                weights_list.append(weight)
                bias_list.append(bias)
            return weights_list, bias_list

        dilate_weights, dilate_biases = get_parameters_list(self, 'dilated')
        skip_weights, skip_biases = get_parameters_list(self, 'skip_out')
        res_weights, res_biases = get_parameters_list(self, 'residual')

        ret_dict = {
            'embedding_prev': None,  # A*R
            'embedding_curr': None,  # A*R
            'conv_out_weight': self.relu_out.weight.data,  # A*S
            'conv_end_weight': self.softmax_out.weight.data,  # A*A
            'dilate_weights': dilate_weights,  # list (2*R)XRx2
            'dilate_biases': dilate_biases,  # list tensor 2*R
            'res_weights': res_weights,  # list R*R
            'res_biases': res_biases,  # list R
            'skip_weights': skip_weights,  # list R*R
            'skip_biases': skip_biases,  # list R
            'use_embed_tanh': True,  # bool use embed_tanh or not
            'max_dilation': self.max_dilation
        }
        return ret_dict

    def comput_condition_input(self, c, g):
        B, _, T = c.size()
        if g is not None:
            if self.embed_speakers is not None:
                # (B x 1) -> (B x 1 x gin_channels)
                g = self.embed_speakers(g.view(B, -1))
                # (B x gin_channels x 1)
                g = g.transpose(1, 2)
                assert g.dim() == 3
        g_bct = _expand_global_features(B, T, g, bct=True)

        if c is not None and self.upsample_conv is not None:
            # B x 1 x C x T
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
        condition_list = []
        for f in self.upsample_conv:
            c = f(c)
        condition = None
        for layer in range(self.layers):
            if self.cin_channels > 0:
                c_f = self._modules['c_{}'.format(layer)]
                c = _conv1x1_forward(c_f, c, False)
                condition += c
            if self.gin_channels > 0:
                g_f = self._modules['g_{}'.format(layer)]
                _g = _conv1x1_forward(g_f, g_bct, False)
                condition += _g
            condition_list.append(condition)
        condition_result = torch.stack(condition_list, 0)  # [L,B,2*R,T]
        condition_result = condition_result.permute(2, 1, 0, 3)
        return condition_result
