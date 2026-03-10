import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def extract_cqt_features(audio_path):
    # 1. Load the audio file (resampled to 22050 Hz by default)
    y, sr = librosa.load(audio_path, sr=None)

    # 2. Compute the Constant-Q Transform
    # 36 bins per octave (3 bins per semitone) gives higher resolution for chords
    cqt = librosa.cqt(y, 
                      sr=sr, 
                      fmin=librosa.note_to_hz('C3'), 
                      n_bins=12*3,        # 3 octaves
                      bins_per_octave=12)
    
    # 3. Convert to decibels (log scale) for better feature representation
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    # 4. Visualization
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(cqt_db, sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format="%+2.0f dB")
    plt.title('Constant-Q Power Spectrogram (CQT)')
    plt.tight_layout()
    plt.show()

    return cqt_db

# Example usage:
# features = extract_cqt_features('tinyAAM/annotations/0001_segments.arff')

import pandas as pd
import arff

with open('tinyAAM/annotations/0001_segments.arff', 'r') as f:
    data = arff.load(f)

df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
print(df)
print(df.dtypes)