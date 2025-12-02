

## For a simple model like Logistic Regression, SVM, RandomForest, these are typically sufficient:
1. MFCCs
The single most important feature.
Using 13–20 MFCCs averaged over time often gives 70–90% accuracy alone.

2. Spectral Centroid
Very helpful because bus vs. car differ in brightness.

3. RMS Energy
Adds loudness + engine-stability information.

## Optional
1. Zero-Crossing Rate
Useful, but MFCCs and spectral centroid already carry similar information.

2. Mel Spectrogram
Needed only if:
you plan to use a CNN or
you want image-based classification