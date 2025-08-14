# EE-452 Network Machine Learning Course Project  
**Graph-based EEG Seizure Detection: A Comparative Study**

This repository contains the code, experiments, and report for our EE-452 course project, which investigates **graph neural networks (GNNs)** for EEG seizure detection, representing electrodes as graph nodes and comparing architectures, graph construction, and preprocessing methods to capture complex temporal–spatial patterns.

## Repository Structure
- **report.pdf** – Final project report  
- **README.md** – Project description  
- **code_final.ipynb** – Main notebook for running experiments.  
  - You can change preprocessing techniques, classification models, or graph construction methods by editing the relevant sections.  
  - By default, it uses **STFT** preprocessing, **ChebNet Attention** model, and **Mutual Information (MI)** graph construction.  
- **Description_Methods.ipynb** – Functions and detailed descriptions for all preprocessing, graph construction, and model methods.  
- **GIN_and_GAT_experiments/** – Additional experiments:  
  - `GAT_experiments/` – GAT training notebooks and results  
  - `GIN_experiments/` – GIN training notebooks and results  
- **chebnet_attn_results/** – Saved results from ChebNet Attention model experiments

## Methods Overview
We experimented with:
- **Preprocessing:** FFT, STFT, Wavelet Transform, Bandpower, and combined feature sets.
- **Graph Construction:** Unweighted 10-20 montage, inverse distance, k-NN, mutual information (MI), and correlation-based graphs.
- **Models:**  
  - GCN  
  - ChebNet (standard, with attention, with residuals)  
  - ST-GNN (CNN/LSTM temporal encoder + GCN)  
  - GraphSAGE  
  - GIN  
  - GAT (with and without supernode pooling)

## How to Run
1. Open `code_final.ipynb` in Jupyter or Colab.
2. Install dependencies (see imports in the notebook or your environment requirements).
3. Adjust preprocessing/model/graph parameters as needed.
4. Run all cells to train and evaluate.

## Authors
Iris Yazici, Hasan Said Unal, Semanur Avşar, Ender Dogan Isik
