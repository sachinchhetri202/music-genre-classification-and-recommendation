# music-genre-classification-and-recommendation
This project builds an end-to-end system for automatically classifying music by genre and recommending similar songs. Our approach uses both traditional machine learning models and deep learning architectures to tackle the problem.

## Project Overview

- **Problem Statement:**  
  Create a system that automatically classifies music by genre and then recommends similar tracks based on learned features.

- **Dataset:**  
  We use the **GTZAN Genre Collection** because it is balanced (100 tracks per genre across 10 genres) and small enough for a class project. (The alternative, Free Music Archive, is too large for our purposes.)

- **Components:**  
  1. **Data Preprocessing & Exploration:**  
     - Loading audio files from the GTZAN dataset.
     - Extracting MFCC features and visualizing genre distribution.
  2. **Baseline Models:**  
     - Implementing two baseline models (Random Forest and SVM) using scikit-learn.
  3. **Deep Learning Architectures:**  
     - One deep learning model (a simple CNN built with PyTorch and PyTorch Lightning) is implemented here.
     - One deep learning model (a modified EfficientNet-B3 that accepts single-channel mel-spectrogram inputs) is implemented here.
  4. **Validation & Evaluation:**  
     - Splitting the dataset into training and validation sets.
     - Reporting accuracy, loss, and confusion matrices.

