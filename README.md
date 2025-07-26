# AlzCLIP

AlzCLIP is a contrastive learning-based framework designed to integrate genetic variants (SNPs) and brain imaging features into a shared representation space for Alzheimer's disease (AD) prediction. By combining contrastive pretraining with a voting-based ensemble classifier, AlzCLIP enables robust multi-modal disease prediction and provides insights into genotype–phenotype interactions.

### Key Features
Two-Stage Learning
- Stage 1: Contrastive pretraining for cross-modal representation learning
- Stage 2: Supervised fine-tuning for Alzheimer's classification
  
Flexible Multimodal Inference
- Combined Mode: SNP + MRI for maximum accuracy
- SNP-Only Mode: When genetic data is available but MRI is not
- MRI-Only Mode: When imaging data is available but genetic data is not


## Table of Contents
- [Background](#background)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Input Data](#input-data)
- [Quick Start (Basic Usage)](#quick-start-basic-usage)
- [Running Details](#running-details)
  - [Training AlzCLIP](#training-alzclip)
  - [Ensemble Voting Inference](#ensemble-voting-inference)
  - [Tutorial Notebook](#tutorial-notebook)
- [Output](#output)
- [Contact](#contact)

## Background
Alzheimer’s disease (AD) is a complex neurodegenerative disorder driven by genetic predisposition and brain structural changes.
Traditional single-modality approaches, relying solely on genetic or imaging features, often fall short in prediction accuracy and biological interpretability.

AlzCLIP addresses this gap by:
- Learning a unified latent space through contrastive learning that aligns SNP and MRI features.
- Enhancing disease classification through a voting ensemble integrating SVM, Random Forest, and XGBoost classifiers.
- Providing interpretable multi-modal feature embeddings for downstream analysis.

## Getting Started
### Prerequisites
You need Python 3.7+ and the following libraries:
* numpy
* pandas
* scipy
* scikit-learn
* torch
* torchvision
* tqdm
* shap
* matplotlib (optional for visualization)

You can install all required packages via:
```python
pip install -r requirements.txt
```
or using conda:
```python
conda env create -f environment.yaml
```

### Input Data
AlzCLIP expects:
* SNP feature matrix per subject
* MRI ROI feature matrix per subject
* Labels
Note: Subjects must be aligned across SNP and imaging files.

To prepare your dataset:
```python
python dataset_making.py \
    --snp_path ./SNP_data/ \
    --img_path ./MRI_features.csv \
    --output_path ./my_dataset \
    --label_type binary          # or 'multiclass' for multiple categories
```

### Quick Start (Basic Usage)
Clone the repository:
```python
git clone https://github.com/QSong-github/AlzCLIP
```

Build the dataset:
```python
cd /path/to/data
python dataset_making.py
```

Train the model:
Run the complete training pipeline:
```python
python main.py --pretrain_epochs 100 --finetune_epochs 50 --batch_size 128
```

Run inference with ensemble voting:
```python
python infer.py 
```

### Running Details
#### Training AlzCLIP
Run `main.py` to:
* Pretrain embeddings using contrastive loss
* Fine-tune embeddings for classification using cross-entropy loss
```python
python main.py \
    --pretrain_epochs 100 \      # Contrastive learning epochs
    --finetune_epochs 50 \       # Supervised learning epochs
    --batch_size 128 \           # Training batch size
    --pretrain_lr 0.001 \        # Pretraining learning rate
    --finetune_lr 0.0001 \       # Fine-tuning learning rate
    --save_path ./save           # Model save directory
    --dataset_path ./snp_lbl_img_dataset_ova  # Dataset path (default)
```

#### Ensemble Voting Inference
Run `infer.py` to:
* Extract SNP and MRI embeddings from trained model
* Train SVM, Random Forest, and XGBoost classifiers
* Perform soft voting (averaging predicted probabilities)
* Output performance metrics 


### Output
After inference, AlzCLIP generates:
* `final_results.npy`: Complete prediction results with probabilities
* `ensemble_classifiers.pkl`: Trained ensemble models
* `embeddings_data.npy`: Extracted SNP and MRI embeddings



### Citation
If you use AlzCLIP in your research, please cite:
`Integrating Genetic and Imaging Data for Alzheimer’s Disease Diagnosis and Interpretation'


   
