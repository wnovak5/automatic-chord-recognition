# **Automatic Chord Recognition**
## **Contributors**: Edward Anderson, Sam Kunitz-Levy, Will Novak

Deep Learning final project to transcribe chords from provided audio files.

## Project Proposal

### Automatic Chord Recognition
### Neural Networks for Musical Chord Recognition

Edward Anderson (`eca4zm`), Will Novak (`dcq8fz`), Sam Kunitz-Levy (`jhb7ek`)

### Motivation

We are interested in building a model that takes in audio files and outputs the sequence of chords that are present in the audio. This is an application of deep learning to recognize musical chords from audio files. Being able to automatically recognize chords from audio has various applications such as music transcription, music information retrieval, and aiding musicians in learning and practicing certain songs.

### Dataset

We are hoping to use one of these datasets for training the model. If we have enough time, we might be able to train on both datasets.

**AAM: Artificial Audio Multitracks Dataset**  
https://zenodo.org/records/5794629

- 3,000 tracks of about 2 minutes each
- 100+ hours of audio
- Synthetically generated audio tracks with chord annotations
- Chord annotations are in `.arff` format
- Created for the purpose of training deep learning models

**GuitarSet**  
https://zenodo.org/records/3371780

- Audio and chord annotations across several songs played in every key at different tempos
- Includes 12-bar blues, Pachelbel's Canon, and Autumn Leaves
- Audio files total about 3 hours of play time, including chords and melody for guitar

### Related Work

**Template Based Chord Recognition**  
This approach to chord recognition compares audio features to predefined chord templates. Audio signals are divided into frames and converted into chroma features, which are 12-dimensional vectors representing the energy of each pitch class (`C` through `B`). These chroma vectors are compared to templates of major and minor chords, and the chord template with the highest similarity is assigned to each frame.

**Beatles Collection**  
This notebook applies chord recognition methods to a set of Beatles songs to compare how different feature representations and recognition approaches perform. The experiments conducted by the researchers show that chord recognition depends heavily on chroma representation, temporal smoothing, and simplification of complex chords into major/minor triads.

### Technical Plan

**Inputs**  
The inputs of the task are audio files. These audio files will be converted to spectrograms, which highlight the energies at different pitches on a log scale. The spectrogram can then be converted to a chromagram, which buckets the pitches and shows the relative energy for each pitch. The combination of pitches defines the chord.

**Outputs**  
The output of the model will be chord annotations.

**Models**  
We plan to use a CNN/RNN architecture.

**Loss Functions**  
We plan to use cross-entropy loss since the outputs will be softmax probabilities for each of the chord labels.

### Evaluation Plan

We would like to train the model using the synthetic AAM dataset.

- First, we can evaluate it on a test set of other AAM songs
- We would also like to see how well the model generalizes to pop songs
- If time allows, we may try to incorporate the GuitarSet data as well for training to see if it helps

### References

- Ostermann, F., Vatolkin, I. & Ebeling, M. *AAM: a dataset of Artificial Audio Multitracks for diverse music information retrieval tasks.* J AUDIO SPEECH MUSIC PROC. 2023, 13 (2023). https://doi.org/10.1186/s13636-023-00278-7
- Q. Xi, R. Bittner, J. Pauwels, X. Ye, and J. P. Bello, *Guitarset: A Dataset for Guitar Transcription*, in 19th International Society for Music Information Retrieval Conference, Paris, France, Sept. 2018.
- Muller, M. (2021). *Fundamentals of Music Processing - Using Python and Jupyter Notebooks.* 2nd edition, Springer Verlag.

Original proposal link: https://docs.google.com/document/d/1ytM4tWTwF1X2r1UOGdtZeUIflqbwEv0d2NdekS39iXM/edit?tab=t.0
