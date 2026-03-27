# EEG Classification for Multiple Sclerosis (MS) Detection
This project explores the feasibility of classifying Multiple Sclerosis (MS) patients using 5-minute resting-state EEG data. The goal was to build a machine learning pipeline capable of distinguishing between MS patients and healthy controls based on frequency-based analysis.

## Project Overview
The study involved a small-scale dataset of 10 subjects:

- 5 MS Patients

- 5 Healthy Controls

Given the limited number of subjects, the signals were windowed to increase the number of samples available for training and testing.






## Methodology
### 1. Signal Processing & Feature Extraction
The raw EEG data was processed by calculating the Band Power for every channel across specific frequency ranges:

- Delta: 0.5 – 4 Hz

- Theta: 4 – 8 Hz

- Alpha: 8 – 16 Hz

- Beta: 16 – 31 Hz

- Gamma: 31 – 50 Hz

> [!NOTE]
> The signal processing method used in this study is the same as the signal processing method used in this study by Sasmaz et al. "A simplified method for relapsing-remitting multiple sclerosis detection: Insights from resting EEG signals".

### 2. Machine Learning Pipeline
We implemented and compared several classification algorithms to identify the best performer for this specific signal type:

- LDA (Linear Discriminant Analysis)

- SVM (Support Vector Machine)

- Random Forest

- Logistic Regression

### Data Split 
- 80% of the windowed data was used for training.

- 20% of the data was reserved for testing and validation.

## File Structure
### File	Description
- oop_channel.py	Handles data ingestion, windowing logic, and the calculation of frequency band powers.
- MS_eeg_detection_model.ipynb	Contains the machine learning implementation, including model training, the 80/20 split, and performance evaluation.
Requirements

> [!IMPORTANT]
> To run these scripts, you will need the following Python libraries:


- numpy

- pandas

- scikit-learn

- scipy (for signal processing)


> [!NOTE]
> This research was ultimately discontinued due to an insufficient amount of data to achieve statistically significant or generalizable results. It remains a proof-of concept for the processing pipeline.

## Conclusion
While the machine learning models were successfully implemented, the sample size (10 subjects) proved too small for a robust clinical classification model. Future iterations of this study would require a significantly larger dataset to validate the efficacy of these EEG biomarkers.

> The study mentioned: https://www.sciencedirect.com/science/article/pii/S0010482524008138
