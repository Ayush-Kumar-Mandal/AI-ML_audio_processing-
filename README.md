# AI-ML_audio_processing-
## This project focuses on detecting emotions from speech using audio signals. The system classifies audio into emotions like calm, happy, sad, angry, fearful, disgust, surprised, and neutral using machine learning and deep learning models.
# Project Description
This project aims to classify human emotions from speech audio samples using a 1D Convolutional Neural Network (CNN). The model is trained and tested on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which contains recordings labeled with different emotional states.
The goal is to accurately recognize and categorize emotions such as angry, calm, disgust, fearful, happy, neutral, sad, and surprised, which can be used in emotion-aware systems like virtual assistants, mental health support systems, or call center monitoring.
# Preprocessing Methodology
#### Filename Metadata Extraction
Audio filenames are parsed to extract emotion ID and other metadata
#### Feature Extraction
Extracted MFCC (Mel Frequency Cepstral Coefficients) features.
40 MFCCs per frame were averaged over time to generate a 1D feature vector per sample.
Final input shape per sample: (174, 40) → Reshaped for 1D CNN input: (174, 40)
#### Label Encoding
Converted emotion labels to numeric form using LabelEncoder.
#### Data Splitting & Scaling
Used train-test split with stratified sampling to maintain class balance.
Features normalized using StandardScaler
#### Model Architecture: 1D CNN
Loss Function: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy
Epochs: 100
#### Evaluation Metrics:
Accuracy
F1-Score (Macro and per class)
Confusion Matrix

| Metric                | Value |
| --------------------- | ----- |
| Accuracy              | 0.80  |
| Macro Avg F1-Score    | 0.79  |
| Weighted Avg F1-Score | 0.80  |

#### Classification Report

| Emotion   | Precision | Recall | F1-score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Angry     | 0.81      | 0.79   | 0.80     | 80      |
| Calm      | 0.80      | 0.65   | 0.72     | 37      |
| Disgust   | 0.75      | 0.84   | 0.79     | 74      |
| Fearful   | 0.79      | 0.88   | 0.83     | 67      |
| Happy     | 0.76      | 0.98   | 0.86     | 65      |
| Neutral   | 0.85      | 0.67   | 0.75     | 49      |
| Sad       | 0.93      | 0.71   | 0.81     | 79      |
| Surprised | 0.79      | 0.82   | 0.80     | 40      |

