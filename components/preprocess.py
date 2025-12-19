import librosa
import os
from pydub import AudioSegment
import numpy as np
import soundfile as sf

# This file converts the data from source_folder to destination_folder
source_path = "../raw_data"
destination_path = "../processed_data"

def get_wavs(path):
    filenames = os.listdir(path)

    # Separate wav files and non-wav files
    wav_files = set()
    non_wavs = {}
    for filename in filenames:
        if filename.startswith("."): continue   # skip hidden files
        name, ext = os.path.splitext(filename)
        if ext.lower() == ".wav": wav_files.add(name)
        else: non_wavs[name] = ext

    source_wavs = list(map(lambda x: os.path.join(path, x) + ".wav", wav_files))

    # Convert non_wav files that don't have wav alternatives
    for name, ext in {n:e for n,e in non_wavs.items() if n not in wav_files}:
        base = os.path.join(path, name)
        nonwav_path = base + ext
        wav_path = base + ".wav"
        AudioSegment.from_file(nonwav_path).export(wav_path, format="wav")
        source_wavs.append(wav_path)

    return source_wavs

def audio_data(source_folder, new_prefix, destination_folder):
    """
        source_folder: path to the original sounds' directory
        new_prefix: new name for the file, e.g. 'bus' => x_bus.wav
        destination_folder: path to the folder where processed data is put
        return: list of the new recordings, not necessary as they are in the new folder anyway
    """
    os.makedirs(destination_folder, exist_ok=True)

    new_sr = 16000
    audio_data_list = []

    wav_paths = get_wavs(source_folder)

    for i, wav_path in enumerate(wav_paths):
        x, sr = librosa.load(wav_path, sr=None)

        if sr != new_sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=new_sr)  # resample 16000 Hz
            sr = new_sr

        x = x / (np.max(np.abs(x)) + 1e-9)  # normalize

        new_name = f"{i}_{new_prefix}.wav"
        save_path = os.path.join(destination_folder, new_name)

        sf.write(save_path, x, sr)
        audio_data_list.append((x, sr, new_name))

    return audio_data_list

def preprocess(modes=("train", "validation", "test"), classes=("car", "tram")):
    for m in modes:
        for c in classes:
            source_folder = os.path.join(source_path, m, c)
            destination_folder = os.path.join(destination_path, m, c)
            processed = audio_data(source_folder, c, destination_folder)


if __name__ == "__main__":
    preprocess()
