import librosa
import os
from pydub import AudioSegment
import numpy as np
import soundfile as sf

#This file converts the data from data folder to processed_data

def convert_to_wav(path):
    base, ext = os.path.splitext(path)
    if ext.lower() != ".wav":
        wav_path = base + ".wav"
        AudioSegment.from_file(path).export(wav_path, format="wav")
        return wav_path
    return path

def audio_data(folder_path, new_prefix, save_folder):
    """
        folder_path: Reitti kansioon missä alkuperäiset ääni tiedostot
        new_prefix: uusi nimi tiedostolle esim. 'bus' == x_bus.wav
        save_folder: Reitti kansioon mihin käsitellyt tiedostot menee
        return: lista missä uudet äänitteet, ei välttättä tarvita sillä äänitteet uudessa kansiossa anyway
    """
    os.makedirs(save_folder, exist_ok=True)

    new_sr = 16000
    audio_data_list = []

    counter = 1

    for filename in os.listdir(folder_path):
        if filename.startswith("."):
            continue

        path = os.path.join(folder_path, filename)

        wav_path = convert_to_wav(path)

        x, sr = librosa.load(wav_path, sr=None)

        if sr != new_sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=new_sr) # resample 16000 Hz
            sr = new_sr

        x = x / (np.max(np.abs(x)) + 1e-9) # normalisointi

        new_name = f"{counter}_{new_prefix}.wav"
        save_path = os.path.join(save_folder, new_name)

        sf.write(save_path, x, sr)

        audio_data_list.append((x, sr, new_name))
        counter += 1

    return audio_data_list




"""
folder_bus = "data/bus"
folder_car = "data/car"
bus_audio_data = audio_data(folder_bus, "bus", "processed_data/bus")
car_audio_data = audio_data(folder_car, "car", "processed_data/car")
"""




