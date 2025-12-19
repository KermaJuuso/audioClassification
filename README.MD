# Audio Classification
This repository tackles a simple classification problem trying to classify 
cars and trams based on recorded audio samples of them passing. 
It would be simple to add other classes as well. 

To demo the program use the `demo.ipynb` Jupyter notebook.


## Environment setup
To run only the demo, you can take only `numpy`, `librosa`, `pandas`, and `pydub` from the following.

To run all the code in this repository you can create a conda virtual environment with these commands
  ```
  conda create --name audioClassification python=3.11
  conda activate audioClassification
  conda install numpy=2.3
  pip install librosa
  conda install matplotlib
  pip install pandas
  pip install notebook
  pip install torch torchaudio
  conda install pydub
  pip install torchcodec
  conda install ffmpeg
  pip install sounddevice
  pip install soundfile
  ```
\
If you want ot utilize (NVidia) GPU, check [PyTorch website](https://pytorch.org/get-started/locally/) for the installation command.  
- (First uninstall the cpu versions, if you already intalled them: `pip uninstall torch torchaudio`)  
- E.g. for Windows, CUDA13: `pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu130`

This might need the installation of CUDA Toolkit and cuDNN library to work properly ([e.g. tutorial](https://www.digitalocean.com/community/tutorials/install-cuda-cudnn-for-gpu))

## Data
Not every audio file is included in GitHub as there are a lot of them.

Put raw demo files into `demo_data` folder.
One self recorded example sample is already in there.

Expected file structure for other data:

```
raw_data/
  test/
    car/
      car001.wav
      etc...
    tram/
  train/
    car/
    tram/
  validation/
```

Preprocessing makes the following from that:

```        
processed_data/
  test/
    car/
      1_car.wav
      etc...
    tram/
  train/
    car/
    tram/
  validation/
```

## Extras

For a simple model like Logistic Regression, SVM, RandomForest, these are typically sufficient:

1. MFCCs
   The single most important feature.
   Using 13–20 MFCCs averaged over time often gives 70–90% accuracy alone.

2. Spectral Centroid
   Very helpful because bus vs. car differ in brightness.

3. RMS Energy
   Adds loudness + engine-stability information.

Optional
1. Zero-Crossing Rate: Useful, but MFCCs and spectral centroid already carry similar information.

2. Mel Spectrogram:
   Needed only if
   you plan to use a CNN or
   you want image-based classification

https://www.geeksforgeeks.org/nlp/mel-frequency-cepstral-coefficients-mfcc-for-speech-recognition/

