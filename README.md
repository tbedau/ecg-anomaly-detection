# ECG Anomaly Detection

A deep learning approach for automated detection of cardiac anomalies in electrocardiogram (ECG) recordings. This project implements a 1D convolutional neural network that classifies heartbeat segments from single-lead ECG signals into normal and abnormal categories.

## Dataset

This project uses the MIT-BIH Arrhythmia Database, which contains 48 half-hour excerpts of two-channel ambulatory ECG recordings from 47 subjects studied at Boston's Beth Israel Hospital. The dataset includes five heartbeat classes following the AAMI standard:

| Class | Label | Count |
|-------|-------|-------|
| N | Normal | 82,771 |
| S | Supraventricular | 2,781 |
| V | Ventricular | 7,012 |
| F | Fusion | 803 |
| Q | Unknown | 8,039 |

## Results

Our model achieves the following performance on the test set:

| Metric | Score |
|--------|-------|
| Accuracy | 97.3% |
| F1 (macro) | 0.947 |
| Precision (macro) | 0.952 |
| Recall (macro) | 0.943 |

## Requirements

- Python 3.10+

## License

MIT
