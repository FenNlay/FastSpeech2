import torch
import numpy as np
import librosa.util as librosa_util
from scipy.signal import get_window


def window_sumsquare(
    window,
    n_frames,
    hop_length,
    win_length,
    n_fft,
    dtype=np.float32,
    norm=None,
):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    # Set the window length to n_fft if not provided
    if win_length is None:
        win_length = n_fft

    # Calculate the total output length
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Get and normalize the window function, then square it for sum-squared calculation
    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Apply the squared window across frames
    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x

# Adjustment suggestions:
# 1. Changing `hop_length` changes the overlap between frames: smaller values give more overlap.
# 2. Modifying `norm` (e.g., 'l1', 'l2', 'max') will affect how the window function is normalized.
# 3. Adjusting `n_frames` changes the overall length of the output envelope.

def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    # Initialize random phase angles
    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))

    # Perform initial inverse STFT
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    # Iteratively refine the phase
    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal

# Adjustment suggestions:
# 1. Increasing `n_iters` refines the phase, improving audio quality, but increases computation time.
# 2. The quality of the reconstruction depends heavily on the STFT and ISTFT implementations in `stft_fn`.

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

# Adjustment suggestions:
# 1. Increasing `C` increases the compression, making quieter parts louder relative to louder parts.
# 2. Adjusting `clip_val` helps avoid numerical issues with very low amplitude values.

def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

# Adjustment suggestions:
# 1. `C` should match the value used in `dynamic_range_compression` to restore original scaling.
# 2. Using a different `C` could result in an unintended scaling factor after decompression.
