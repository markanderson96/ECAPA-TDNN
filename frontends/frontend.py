import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from frontends.specaugment import SpecAugment
from frontends.leaf import PCEN as PCENLayer
from frontends.TDFbanks import TDFbanks
from frontends.OG_leaf import Leaf
from frontends.efficientleaf import EfficientLeaf
import frontends.leaf_audio_pytorch.frontend as leaf_audio

class Frontend(nn.Module):
    def __init__(self, conf):
        # empty list for frontend layers,
        # all the layers are already defined but depending on
        # what is specified in the config,
        # only some of them will be loaded into the model
        super(Frontend, self).__init__()
        input_channels = 1 # annoying workaround for STRF
        _frontend_layers = []
        if conf.features.frontend == 'logmel':
            if conf.features.spectaugment:
                _frontend_layers.append(
                    torchaudio.transforms.Spectrogram(
                        n_fft=conf.features.n_fft,
                        hop_length=conf.features.hop_length,
                        win_length=conf.features.n_fft,
                        center=True,
                        pad_mode='reflect',
                        power=None,
                        return_complex=True
                    )
                )
                _frontend_layers.append(
                    SpecAugment(
                        time_mask_param=conf.features.augmetation.time_mask,
                        freq_mask_param=conf.features.augmetation.freq_mask,
                        n_fft=conf.features.n_fft,
                        rate=1.2
                    )
                )
            else:
                _frontend_layers.append(
                    torchaudio.transforms.Spectrogram(
                        n_fft=conf.features.n_fft,
                        hop_length=conf.features.hop_length,
                        win_length=conf.features.n_fft,
                        center=True,
                        pad_mode='reflect',
                    )
                )
            _frontend_layers.append(
                nn.Sequential(
                    torchaudio.transforms.MelScale(
                        sample_rate=conf.features.sample_rate,
                        n_stft=conf.features.n_fft//2+1,
                        n_mels=conf.features.n_mels,
                        f_min=conf.features.f_min,
                        f_max=conf.features.f_max,
                    ),
                    torchaudio.transforms.AmplitudeToDB()
                )
            )

        elif conf.features.frontend == 'pcen':
            if conf.features.spectaugment:
                _frontend_layers.append(
                    torchaudio.transforms.Spectrogram(
                        n_fft=conf.features.n_fft,
                        hop_length=conf.features.hop_length,
                        win_length=conf.features.n_fft,
                        center=True,
                        pad_mode='reflect',
                        power=None,
                        return_complex=True
                    )
                )
                _frontend_layers.append(
                    SpecAugment(
                        time_mask_param=conf.features.augmentation.time_mask,
                        freq_mask_param=conf.features.augmentation.freq_mask,
                        n_fft=conf.features.n_fft,
                        rate=1.2
                    )
                )
            else:
                _frontend_layers.append(
                    torchaudio.transforms.Spectrogram(
                        n_fft=conf.features.n_fft,
                        hop_length=conf.features.hop_length,
                        win_length=conf.features.n_fft,
                        center=True,
                        pad_mode='reflect',
                    )
                )
            _frontend_layers.append(
                PCENLayer(
                    num_bands=conf.features.n_mels,
                    s=0.05,
                    alpha=0.8,
                    delta=10.,
                    r=2.,

                )
            )

        elif conf.features.frontend == 'td':
            _frontend_layers.append(
                TDFbanks(
                    mode='learnfbanks',
                    nfilters=conf.features.n_mels,
                    n_fft=conf.features.n_fft,
                    samplerate=conf.features.sample_rate,
                    f_min=conf.features.f_min,
                    f_max=conf.features.f_max,
                    wlen=64,
                    wstride=8,
                )
            )

        elif conf.features.frontend == 'leaf':
            _frontend_layers.append(
                Leaf(
                    n_filters=conf.features.n_mels,
                    min_freq=conf.features.f_min,
                    max_freq=conf.features.f_max,
                    sample_rate=conf.features.sample_rate,
                    window_len=64.,
                    window_stride=10.,
                    sort_filters=True
                )
            )
        elif conf.features.frontend == 'efficientleaf':
            _frontend_layers.append(
                EfficientLeaf(
                    n_filters=conf.features.n_mels,
                    num_groups=8,
                    min_freq=conf.features.f_min,
                    max_freq=conf.features.f_max,
                    sample_rate=conf.features.sample_rate,
                    window_len=64.,
                    window_stride=10.,
                    compression="pcen",
                    init_filter=conf.features.init_filter
                )
            )
        else:
            raise Exception("Must specify a valid frontend in config/config.yaml")
        self.frontend_layers = nn.Sequential(*_frontend_layers)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, inputs):
        output = self.frontend_layers(inputs)
        #output = output.unsqueeze(1)
        #output = self.bn(output)
        #output = output[:,0,:,:]
        return torch.permute(output.squeeze(), (0, 2, 1))
