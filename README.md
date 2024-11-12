# Cryptocurrency Price Prediction

This project focuses on predicting cryptocurrency prices using advanced deep learning models, including LSTM and CNN, combined with multi-head attention and layer normalization techniques. The model aims to achieve higher accuracy in forecasting by capturing complex temporal patterns in cryptocurrency data.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Setup and Usage](#setup-and-usage)
- [Training Scripts](#training-scripts)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The main objective of this project is to predict future cryptocurrency prices by developing a deep learning model capable of handling high variability and volatility. The project leverages LSTM and CNN architectures to capture both sequential and spatial patterns, enhanced by multi-head attention mechanisms for improved accuracy.

## Features

- **LSTM Model**: Captures sequential dependencies in time-series data.
- **CNN Model**: Extracts local features from input data, capturing spatial dependencies.
- **Multi-Head Attention**: Enhances the model’s ability to focus on relevant data patterns.
- **Layer Normalization**: Improves convergence and stability in deep networks.
- **Scalability**: Can be trained on CPU and GPU, with SLURM scripts provided for large-scale environments.

## Requirements

The project relies on the following Python packages:

```plaintext
- joblib
- keras
- matplotlib
- numpy
- pandas
- requests
- scikit-learn
- tensorflow
- tensorboard
- streamlit
