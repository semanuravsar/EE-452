# EE-452 Network Machine Learning Course Project: Graph-based EEG Seizure Detection: A Comparative Study

This folder contains our course project on graph-based approaches for EEG seizure classification, comparing different graph neural network architectures.


```bash
.
├── report.pdf                          # Final project report
├── README.md                           
├── code_final.ipynb                    # Code to run. You may change to any preprocessing technique, classification model or graph construction tecnique by changing the appropriate lines. If you run it without any changes, it will use STFT, ChebNET Attn, and MI graph.
├── Description_Methods.ipynb          # Functions and descriptions for all methods
│
├── GIN_and_GAT_experiments/
│   ├── GAT_experiments/                # Contains the GAT training notebook and experiments
│   └── GIN_experiments/                # Contains the GIN training notebook and experiments
├── chebnet_attn_results/           
