import librosa
import os
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import json
import pandas as pd


def extract_features(folder_path, label):
    rows = []

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
        row = {"filename": filename,
               "label": label}
        
        for i in range(20):
            row[f"mfcc_{i+1}_mean"] = mfcc_mean[i]
            row[f"mfcc_{i+1}_std"]  = mfcc_std[i]

       
        row["scent_mean"] = scent_mean
        row["scent_std"] = scent_std
        row["rms_mean"] = rms_mean
        row["rms_std"] = rms_std

        rows.append(row)

    return rows

bus_rows = extract_features("processed_data/bus", label="bus")
car_rows = extract_features("processed_data/car", label="car")

df = pd.DataFrame(bus_rows + car_rows)
df.to_csv("features.csv", index=False)