import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn

from audio.audio_processing import (
    dynamic_range_compression,
    dynamic_range_decompression,
    window_sumsquare,
)


class STFT(torch.nn.Module):
    """
    adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft
    Performs Short-Time Fourier Transform (STFT) and Inverse STFT (ISTFT).
    """

    def __init__(self, filter_length, hop_length, win_length, window="hann"):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None

        # Scale factor to adjust magnitude
        scale = self.filter_length / self.hop_length

        # Create Fourier basis for STFT
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        
        # Keep only unique components up to the Nyquist frequency
        cutoff = int((self.filter_length / 2 + 1))    # Keep half for real-valued input
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        # Transform basis to PyTorch tensors for convolution
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert filter_length >= win_length
            # Apply and zero-pad the window function to match filter length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # Window the Fourier bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        # Register as non-trainable buffers for forward and inverse basis
        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        """Performs STFT on the input signal."""
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input data at the edges
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        # Perform convolution to obtain the forward transform
        forward_transform = F.conv1d(
            input_data.cuda(),
            torch.autograd.Variable(self.forward_basis, requires_grad=False).cuda(),
            stride=self.hop_length,
            padding=0,
        ).cpu()

        # Split into real and imaginary parts
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        # Compute magnitude and phase from real and imaginary parts
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        """Performs inverse STFT to reconstruct the waveform."""
        # Recombine magnitude and phase
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        # Perform inverse convolution to reconstruct time-domain signal
        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        # Correct for modulation effects if windowing was applied
        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            
            # Adjust for non-zero areas to avoid division errors
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # Scale by the ratio of filter to hop length
            inverse_transform *= float(self.filter_length) / self.hop_length

        # Remove padding applied earlier during forward STFT
        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data):
        """Computes forward and inverse STFT to reconstruct input signal."""
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(torch.nn.Module):
    """Tacotron STFT wrapper with mel-spectrogram computation for speech synthesis."""
    def __init__(
        self,
        filter_length,
        hop_length,
        win_length,
        n_mel_channels,
        sampling_rate,
        mel_fmin,
        mel_fmax,
    ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels    # Number of mel filter banks
        self.sampling_rate = sampling_rate      # Sampling rate of the audio
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        
        # Create mel filter bank with librosa
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes):
        """Apply dynamic range compression for normalization."""
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        """Undo dynamic range compression."""
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        
        # Ensure input is within expected range
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        # Compute magnitude and phase with STFT
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        
        # Project magnitude spectrogram onto mel filter bank
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        
        # Compute energy for each time frame
        energy = torch.norm(magnitudes, dim=1)

        return mel_output, energy

# Adjustments and tuning:
# 1. In `STFT`, adjusting `filter_length` and `hop_length` will alter the time-frequency resolution.
# 2. In `TacotronSTFT`, changing `n_mel_channels`, `mel_fmin`, or `mel_fmax` will affect the mel-spectrogram’s resolution and frequency range.
# 3. Increasing `win_length` (up to `filter_length`) improves frequency resolution but decreases time resolution.
# 4. The `mel_spectrogram` method’s energy output could be adjusted or weighted for different audio features.
