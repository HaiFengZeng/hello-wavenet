# coding: utf-8
from __future__ import with_statement, print_function, absolute_import


def wavenet(out_channels=256,
            layers=20,
            stacks=2,
            residual_channels=512,
            gate_channels=512,
            skip_out_channels=512,
            cin_channels=-1,
            gin_channels=-1,
            weight_normalization=True,
            dropout=1 - 0.95,
            kernel_size=3,
            n_speakers=None,
            upsample_conditional_features=False,
            upsample_scales=[16, 16],
            freq_axis_kernel_size=3,
            scalar_input=False,
            use_speaker_embedding=True,
            ):
    from wavenet_vocoder import WaveNet

    model = WaveNet(out_channels=out_channels, layers=layers, stacks=stacks,
                    residual_channels=residual_channels,
                    gate_channels=gate_channels,
                    skip_out_channels=skip_out_channels,
                    kernel_size=kernel_size, dropout=dropout,
                    weight_normalization=weight_normalization,
                    cin_channels=cin_channels, gin_channels=gin_channels,
                    n_speakers=n_speakers,
                    upsample_conditional_features=upsample_conditional_features,
                    upsample_scales=upsample_scales,
                    freq_axis_kernel_size=freq_axis_kernel_size,
                    scalar_input=scalar_input,
                    use_speaker_embedding=use_speaker_embedding,
                    )

    return model


def student_wavenet(out_channels=2,
                    layers=20,
                    stacks=2,
                    residual_channels=64,
                    iaf_layer_sizes=[10, 20, 30],
                    gate_channels=64,
                    kernel_size=3, dropout=1 - 0.95,
                    cin_channels=-1, gin_channels=-1, n_speakers=None,
                    weight_normalization=True,
                    upsample_conditional_features=False,
                    upsample_scales=None,
                    freq_axis_kernel_size=3,
                    scalar_input=False,
                    use_speaker_embedding=True
                    ):
    from wavenet_vocoder import StudentWaveNet

    model = StudentWaveNet(out_channels=out_channels,
                           layers=layers, stacks=stacks,
                           residual_channels=residual_channels,
                           iaf_layer_sizes=iaf_layer_sizes, gate_channels=gate_channels, kernel_size=kernel_size,
                           dropout=dropout,
                           cin_channels=cin_channels, gin_channels=gin_channels,
                           n_speakers=n_speakers,
                           upsample_conditional_features=upsample_conditional_features,
                           upsample_scales=upsample_scales,
                           freq_axis_kernel_size=freq_axis_kernel_size,
                           scalar_input=scalar_input,
                           use_speaker_embedding=use_speaker_embedding)
    return model


def nv_wavenet(out_channels=256,
               layers=24,
               stacks=4,
               residual_channels=64,
               gate_channels=128,
               skip_out_channels=256,
               cin_channels=-1,
               gin_channels=-1,
               weight_normalization=True,
               dropout=1 - 0.95,
               kernel_size=2,
               n_speakers=None,
               upsample_conditional_features=False,
               upsample_scales=[4, 4, 4, 4],
               freq_axis_kernel_size=3,
               scalar_input=False,
               use_speaker_embedding=True,
               ):
    from wavenet_vocoder import NvWaveNet
    assert residual_channels*2 == gate_channels
    model = NvWaveNet(out_channels=out_channels,
                      layers=layers,
                      stacks=stacks,
                      residual_channels=residual_channels,
                      gate_channels=gate_channels,
                      skip_out_channels=skip_out_channels,
                      kernel_size=kernel_size,
                      dropout=dropout,
                      weight_normalization=weight_normalization,
                      cin_channels=cin_channels,
                      gin_channels=gin_channels,
                      n_speakers=n_speakers,
                      upsample_conditional_features=upsample_conditional_features,
                      upsample_scales=upsample_scales,
                      freq_axis_kernel_size=freq_axis_kernel_size,
                      scalar_input=scalar_input,
                      use_speaker_embedding=use_speaker_embedding,)

    return model
