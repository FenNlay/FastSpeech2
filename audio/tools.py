import torch
import numpy as np
from scipy.io.wavfile import write

from audio.audio_processing import griffin_lim


def get_mel_from_wav(audio, _stft):
    # Convert audio to a PyTorch tensor, ensure values are clipped between -1 and 1, and add batch dimension
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)

    # Compute mel-spectrogram and energy using the STFT object
    melspec, energy = _stft.mel_spectrogram(audio)

    # Remove batch dimension and convert to numpy arrays
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)

    return melspec, energy

# Adjustments:
# 1. Adjusting the `_stft` parameters like `n_mel_channels`, `mel_fmin`, and `mel_fmax` will change the
#    frequency range and resolution of the mel-spectrogram.
# 2. Increasing the sampling rate in `_stft` will affect the detail captured in `melspec`.
# 3. Adding preprocessing, such as normalization, can improve robustness when dealing with varied input audio levels.

def inv_mel_spec(mel, out_filename, _stft, griffin_iters=60):
    # Prepare mel-spectrogram for inversion by adding batch dimension
    mel = torch.stack([mel])

    # Apply spectral de-normalization to retrieve original scale
    mel_decompress = _stft.spectral_de_normalize(mel)

    # Transpose for correct shape and move to CPU for further processing
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()

    # Convert mel-spectrogram to linear spectrogram
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)

    # Reformat to (batch, frequency, time) for Griffin-Lim algorithm
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    # Use Griffin-Lim to estimate the phase and reconstruct waveform from magnitude
    audio = griffin_lim(
        torch.autograd.Variable(spec_from_mel[:, :, :-1]), _stft._stft_fn, griffin_iters
    )

    # Finalize audio output, convert to numpy, and save to specified file
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, _stft.sampling_rate, audio)

# Adjustments:
# 1. Changing `griffin_iters` affects reconstruction quality: more iterations improve quality but increase computation time.
# 2. Adjusting `spec_from_mel_scaling` can enhance output volume but may introduce distortion if too high.
# 3. Modifying `_stft.mel_basis` parameters (e.g., `mel_fmin`, `mel_fmax`) changes the mel-spectrogram frequency range
#    and thus the reconstructed waveform's quality and characteristics.
