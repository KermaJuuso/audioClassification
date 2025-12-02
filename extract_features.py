import librosa
import os
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import json


def extract_features(folder_path):
    for filename in os.listdir(folder_path):
        if filename.startswith("."):
            continue

        path = os.path.join(folder_path, filename)
        x, sr = librosa.load(path, sr=None)


        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20)
        mfcc_mean = mfcc.mean(axis=1) # Näitä tulee n_mfcc määrään mukaa
        mfcc_std = mfcc.std(axis=1)

        # Extract Spectral Centroid
        scent = librosa.feature.spectral_centroid(y=x, sr=sr)
        scent_mean = scent.mean()
        scent_std = scent.std()

        # Extract RMS energy
        rms = librosa.feature.rms(y=x)
        rms_mean = rms.mean()
        rms_std = rms.std()

        # Sitte jotenki jonkunlaisee muotoo et pystyy kouluttaa, vaikka CSV tai pickle tiedostoo tai numpy tiedosto