import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    
    for speaker in tqdm(os.listdir(in_dir)):
        if os.path.isdir(os.path.join(in_dir, speaker)):
            for file_name in os.listdir(os.path.join(in_dir, speaker)):
                if file_name.endswith(".wav"):
                    base_name = file_name[:-4]
                    
                    # Construct paths for the text and wav files
                    text_path = os.path.join(in_dir, speaker, f"{base_name}.txt")
                    wav_path = os.path.join(in_dir, speaker, f"{base_name}.wav")
                    
                    # Read and clean text
                    with open(text_path, "r") as f:
                        text = f.readline().strip("\n")
                    text = _clean_text(text, cleaners)
                    
                    # Create output directory for the speaker
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    
                    # Process the audio
                    wav, _ = librosa.load(wav_path, sr=sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    wavfile.write(
                        os.path.join(out_dir, speaker, f"{base_name}.wav"),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                    
                    # Write the cleaned text to the label file
                    with open(
                        os.path.join(out_dir, speaker, f"{base_name}.lab"),
                        "w",
                    ) as f1:
                        f1.write(text)