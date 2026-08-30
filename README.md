# Breast Cancer Classification using 1D CNN in PyTorch

This project implements a 1D Convolutional Neural Network (Conv1D) using PyTorch to classify breast mass samples as **Malignant** or **Benign** based on continuous morphological features.

## Project Overview
- **Dataset:** Kaggle Breast Cancer Dataset (30 continuous numerical features)
- **Model Architecture:** 1D CNN (`nn.Conv1d`, `nn.MaxPool1d`, `nn.Linear`)
- **Framework:** PyTorch, Scikit-Learn, Matplotlib
- **Accuracy:** ~95.6%

## Features
- Automated dataset extraction and feature standardization
- Loss and accuracy visualization across training epochs

## Setup & Running
1. Install dependencies:
   ```bash
   pip install torch pandas scikit-learn matplotlib