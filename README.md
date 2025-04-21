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
     - One deep learning model (a modified EfficientNet-B7 that accepts single-channel mel-spectrogram inputs) is implemented here.
  4. **Validation & Evaluation:**  
     - Splitting the dataset into training and validation sets.
     - Reporting accuracy, loss, and confusion matrices.
  5. **Hyperparameter Tuning & Final CNN**  
     - Optuna‑driven search over learning rate, dropout, filter count, and batch size.  
     - Full‑dataset retrain for **20 epochs** with **EarlyStopping** and a **TQDMProgressBar** for live progress.  
     - Final test‑set performance: **62 % accuracy** | Macro‑F1 ≈ 0.59.  
     - Saves best checkpoint at `notebooks/checkpoints/best_cnn.ckpt`.
  6. **End‑to‑End Inference & Recommendation Demo**  
     - **Preprocessing**: `preprocess_audio()` loads a raw `.au` file and converts it to a fixed‑size (1×128×128) Mel‑spectrogram tensor.  
     - **Embedding extraction**: monkey‑patches `model.get_embedding()` to run the CNN’s conv‑blocks + global‑avg‑pool, returning a 1D feature vector.  
     - **Recommendation**: `predict_and_recommend(sample_clip, k=5)` classifies the clip (via the loaded checkpoint) and computes cosine similarity against an embedding bank to pick the top‑5 closest tracks.  
     - **Demo output**: prints the input filename, predicted genre, and displays a pandas DataFrame with columns `file`, `genre`, and `cosine_dist` for the recommended songs.  

## How to run

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
