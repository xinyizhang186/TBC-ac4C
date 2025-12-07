# TBC-ac4C

This repository contains code for "**TBC-ac4C: Identiﬁcation of N4-acetylcytidine sites by incorporating TCN and BiGRU based on multi-head cross-attention mechanism **.

# 1 Description

TBC-ac4C is a deep learning model for identifying N4-acetylcytidine (ac4C) sites in mRNA, featuring an innovative architecture that integrates a Transformer encoder, parallel Temporal Convolutional Network (TCN) and Bidirectional Gated Recurrent Unit (BiGRU) modules, and a multi-head cross-attention mechanism. The model was trained on a benchmark dataset, and it exhibits outstanding predictive performance and generalization capability on both balanced and imbalanced test sets. Ac4C  modulates the lifecycle dynamics of mRNA and is closely associated with numerous human pathologies, including cancer and metabolic disorders. Consequently, we developed the TBC-ac4C model to enhance the prediction accuracy of ac4C sites and to support the exploration of its translational applications in future medicine.

## 2 Requirements

Before running, please create a new environment using this command:

```bash
conda create -n tbc-ac4C python=3.10
conda activate tbc-ac4C
```

Next, run the following command to install the required packages:

```bash
cd TBC-ac4C
pip install -r requirements.txt --no-cache-dir
```

# 3 Running

*   **`dataset/`**: Contains the required datasets, specifically `iRNA-ac4C` and `Meta-ac4C`.  
*   **`model/`**: Stores pre-trained models ready for inference. It includes `TBC-ac4C.pt`.  
*   **`util/`**: Utility modules.  
    - `data_loader.py`: Handles data loading and preprocessing.  
    - `util_metric.py`: Computes and evaluates model performance metrics.  

In addition, the main scripts and files are as follows:

*   **`requirements.txt`**: Lists all dependencies and their versions for quick environment setup.  
*   **`model.py`**: Defines the TBC-ac4C model architecture.  
*   **`train.py`**: Training script can be run directly to train the model.  
*   **`test.py`**: Testing script can be run directly to evaluate the model and reproduce results.

If you  aim to train the TBC-ac4C model or use a successfully trained model for testing, please run the following code:

```python
python train.py 
python test.py
```

# 4 Predict

To predict ac4C modification sites, prepare the target sequences in FASTA format and set the input file path in `predict.py`. The script will then output the probabilities of 'modified' and 'unmodified' states, as well as the final predictive label for each potential site. For a detailed implementation, please see the example in `predict.ipynb`.



