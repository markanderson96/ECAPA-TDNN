import numpy as np
import torch

def filter_response(model):
    frontend = model.conf.features.frontend
    filterbank = model.frontend.frontend_layers[-1].filterbank
    center_freqs = filterbank.center_freqs
    bandwidths = filterbank.bandwidths
    center_freqs = center_freqs.detach().cpu()
    bandwidths = bandwidths.detach().cpu()
    impulse_responses = gabor_filters(
        1025,
        center_freqs,
        bandwidths
    )

    freq_responses, db = impulse_to_freq(impulse_responses)
    return freq_responses

def impulse_to_freq(impulse_responses):
    freq_responses = torch.abs(torch.fft.fft(impulse_responses))
    freq_responses_roll = freq_responses.roll(freq_responses.shape[1]//2, dims=1)
    freq_responses_roll = freq_responses_roll + 1E-12
    freq_responses_db = 20 * freq_responses_roll.log10()
    freq_responses_db = freq_responses_db.nan_to_num(nan=0., posinf=0., neginf=0.)

    freq_responses = freq_responses.detach().cpu().numpy()
    freq_responses_db = freq_responses_db.detach().cpu().numpy()

    return freq_responses, freq_responses_db

def gabor_filters(
    size: int,
    center_freqs: torch.Tensor,
    sigmas: torch.Tensor
) -> torch.Tensor:
    """
    Calculates a gabor function from given center frequencies and bandwidths that can be used
    as kernel/filters for an 1D convolution
    :param size: kernel/filter size
    :param center_freqs: center frequencies
    :param sigmas: sigmas/bandwidths
    :return: kernel/filter that can be used 1D convolution as tensor
    """
    t = torch.arange(-(size // 2), (size + 1) // 2, device=center_freqs.device)
    denominator = 1. / (np.sqrt(2 * np.pi) * sigmas)
    gaussian = torch.exp(torch.outer(1. / (2. * sigmas**2), -t**2))
    sinusoid = torch.exp(1j * torch.outer(center_freqs, t))
    return denominator[:, np.newaxis] * sinusoid * gaussian
