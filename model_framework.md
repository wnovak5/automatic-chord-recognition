# Automatic Chord Recognition Interview Prep

## 30-Second Pitch

This project is an automatic chord recognition system. It takes song audio plus chord annotations from the AAM dataset, converts each track into 12-bin chroma features, aligns one chord label per frame, and trains a 2-layer bidirectional LSTM to predict chords frame by frame. Then it smooths the frame predictions and merges short segments so the output is a cleaner chord timeline. Inference is packaged as both a CLI and a local Streamlit app.

## Core Facts To Know Cold

- Input features: `12`-dimensional chroma per frame
- Audio preprocessing: mono, `22050 Hz`, hop length `512`
- Model: `2`-layer bidirectional LSTM
- Hidden size: `128` per direction
- Effective recurrent output size per timestep: `256`
- Output classes: `25`
- Vocabulary: `N.C.` + 12 major + 12 minor chords
- Sequence length during training: `128`
- Optimizer: `Adam`
- Loss: weighted cross-entropy
- Padding label ignored with `ignore_index=-100`

## Questions You Will Likely Get

### Why did you choose chroma features?

A strong answer: chord recognition is mostly about harmonic content, and chroma compresses the spectrum into 12 pitch classes, which is a good inductive bias. It reduces input dimensionality and makes the task easier than learning directly from raw waveforms with limited data.

### Why an LSTM?

A strong answer: chords evolve over time, so framewise independent classification loses temporal context. The bidirectional LSTM lets the model use both past and future harmonic context while staying much lighter and simpler than a transformer.

### Why bidirectional?

A strong answer: chord labels often depend on surrounding context, especially near transitions. Bidirectional recurrence improves per-frame labeling because the model can look both backward and forward within the sequence.

### Why slice into sequences of 128 instead of whole songs?

A strong answer: full songs are too variable in length and more memory-expensive. Fixed-length chunks make batching practical, and temporal structure is still preserved with packed padded sequences.

### Why weighted cross-entropy?

A strong answer: chord labels are imbalanced. Common chords dominate, so inverse-frequency class weights reduce bias toward frequent classes and help minority classes contribute more to training.

### What are the weaknesses of this approach?

A strong answer: the vocabulary is simplified to major, minor, and no-chord, so it cannot represent sevenths or more complex harmony. Also, chroma discards some information, and framewise prediction plus postprocessing can still make boundary mistakes.

### Why not use a transformer or CNN?

A strong answer: those are reasonable next steps, but for this dataset and scope I prioritized a model that is interpretable, computationally manageable, and easy to train end-to-end. The LSTM gave temporal modeling without requiring a much larger system.

### What does postprocessing do?

A strong answer: after per-frame prediction, the pipeline applies majority-vote smoothing and merges short segments into neighbors. That removes jitter and makes the output more musically plausible for a chord chart.

## Good Tradeoff Answers

- Why not raw audio: more expressive, but much harder and more data-hungry.
- Why not larger vocabulary: more realistic, but increases class sparsity and makes training harder.
- Why not unidirectional: cheaper, but worse context.
- Why cache features: preprocessing audio repeatedly is expensive; cached `.npz` tracks make training faster and more reproducible.

## If They Push On LSTM Internals

- LSTM has 4 gates: input, forget, cell/update, output
- That is why weights use `4 * hidden_dim`
- With hidden size `128`, each gate stack is `512`
- Because the model has `2` layers and is bidirectional, that structure appears for each layer and direction

## Best "What Would You Improve Next?" Answer

- Expand the label space beyond only major/minor chords
- Compare LSTM against transformer or CRNN baselines
- Add better evaluation on chord-segment quality, not just frame accuracy
- Tune smoothing and sequence length more systematically
- Add key-awareness or beat-synchronous features

## Good Closing Line

I would describe the main engineering decision in this project as choosing a representation and model that matched the musical structure of the task: chroma for harmonic content, an LSTM for temporal context, and lightweight postprocessing to turn noisy frame predictions into usable chord segments.
