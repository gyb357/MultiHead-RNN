# Table of Contents

1. [Introduction](#Introduction)
2. [Architecture Overview](#Architecture-Overview)
3. [Getting Started](#Getting-Started)
   * Check the Running Environment
   * Installation and Dependencies
   * Download Datasets
   * Configuration
   * Training the Model


---


# 📑Introduction

## Multi-Head Neural Networks for Financial Time Series Classification

This repository implements a multi-head architecture for bankruptcy prediction on financial time series. It uses recurrent models—LTC, CfC, LSTM, and GRU—to process multiple financial indicators in parallel, and includes various preprocessing techniques, undersampling methods to address class imbalance, and comprehensive evaluation metrics.


---


# 🔍Architecture Overview

## Multi-Head Architecture

<div align="center">
   <img src="assets/architecture.png" alt="Multi-head architecture schematic" />
</div>


The architecture employs a multi-head design where each financial variable is processed through its own dedicated network branch, with outputs subsequently combined for final classification.

> **Note:** Papers are [here](https://www.mdpi.com/1999-5903/16/3/79).

### 1. Liquid Time-Constant Networks (LTC)

Continuous-time recurrent network with liquid time constants. Uses the [ncps](https://github.com/mlech26l/ncps) library for implementation.

* **Paper:** [https://arxiv.org/pdf/2006.04439](https://arxiv.org/pdf/2006.04439)

### 2. Closed-form Continuous-time Networks (CfC)

Efficient continuous-time networks with closed-form solutions. Uses the [ncps](https://github.com/mlech26l/ncps) library with lecun_tanh activation.

* **Paper:** [https://arxiv.org/pdf/2106.13898](https://arxiv.org/pdf/2106.13898)


### 3. Recurrent Neural Network (RNN)

Recurrent network capturing sequential dependencies via hidden‑state recurrence.

### 4. Long Short-Term Memory (LSTM)

Traditional recurrent network with memory cells.

### 5. Gated Recurrent Unit (GRU)

Simplified recurrent network with gating mechanisms.


---


# 🔨Getting Started

## 1. Check the Running Environment

Verify your PyTorch installation:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## 2. Installation and Dependencies

Clone the repository and install dependencies:

```bash
git clone https://github.com/gyb357/MultiHeadLNN
cd MultiHeadLNN
pip install -r requirements.txt
```

### Required Dependencies

* torch
* pandas
* matplotlib
* scikit-learn
* ncps
* rich
* pyyaml

## 3. Download Datasets

```bash
https://github.com/sowide/multi-head_LSTM_for_bankruptcy-prediction
```

> **Note:** The dataset is under a CC-BY-4.0 license. Please refer to each repository's README.

## 4. Configuration

Modify `./config/configs.yaml` to customize your experiment.

## 5. Training the Model

### Data Preparation

Ensure your dataset is structured as:

```
dataset/{window}_train.csv
ataset/{window}_valid.csv
dataset/{window}_test.csv
```

### Running Training

```bash
python main.py
```

### Results

* Model checkpoints: `result/best_model.pth`
* Experimental results: `result/[Model Name]_[Scaler Name]_[Threshold]_[RNN Hidden Size]_[FC Hidden Size].csv`
* Show plots: `python plot.py`

