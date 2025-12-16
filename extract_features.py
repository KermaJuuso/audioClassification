import librosa
import os
import pandas as pd

source_path = "processed_data"


def extract_mfcc(audio, sr, row):
    # Extract MFCCs
    nmfcc = 20
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=nmfcc)
    mfcc_mean = mfcc.mean(axis=1)  # Näitä tulee n_mfcc määrään mukaa
    mfcc_std = mfcc.std(axis=1)

    # Add to row
    for i in range(nmfcc):
        row[f"mfcc_{i+1}_mean"] = mfcc_mean[i]
        row[f"mfcc_{i+1}_std"] = mfcc_std[i]


def extract_spec_centroid(audio, sr, row):
    # Extract Spectral Centroid
    scent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    scent_mean = scent.mean()
    scent_std = scent.std()

    # Add to row
    row["scent_mean"] = scent_mean
    row["scent_std"] = scent_std


def extract_spec_bandwidth(audio, sr, row):
    # Extract Spectral Bandwidth
    sband = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    sband_mean = sband.mean()
    sband_std = sband.std()

    # Add to row
    row["sband_mean"] = sband_mean
    row["sband_std"] = sband_std


def extract_rms(audio, row):
    # Extract RMS energy
    rms = librosa.feature.rms(y=audio)
    rms_mean = rms.mean()
    rms_std = rms.std()

    # Add to row
    row["rms_mean"] = rms_mean
    row["rms_std"] = rms_std


def extract_all(folder_path, label):
    rows = []

    for filename in os.listdir(folder_path):
        if filename.startswith("."): continue  # skip hidden files

        # Load the audio sample
        path = os.path.join(folder_path, filename)
        x, sr = librosa.load(path, sr=None)

        # Save the features into a CSV file row by row
        row = {"filename": filename, "label": label}
        extract_mfcc(x, sr, row)
        extract_spec_centroid(x, sr, row)
        extract_spec_bandwidth(x, sr, row)
        extract_rms(x, row)

        rows.append(row)

    return rows


def extract_features(modes=("train", "test"), classes=("car", "tram")):
    for mode in modes:
        rows = []
        for c in classes:
            rows += extract_all(os.path.join(source_path, mode, c), label=c)
        df = pd.DataFrame(data=rows)
        df.to_csv(f'{mode}_features.csv', index=False)


if __name__ == "__main__":
    extract_features(modes=("train", "test"), classes=("car", "tram"))
