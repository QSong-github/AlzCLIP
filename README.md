# AlzCLIP

AlzCLIP is a contrastive learning-based framework designed to integrate genetic variants (SNPs) and brain imaging features into a shared representation space for Alzheimer's disease (AD) prediction. By combining contrastive pretraining with a voting-based ensemble classifier, AlzCLIP enables robust multi-modal disease prediction and provides insights into genotype–phenotype interactions.


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

We provide example datasets:
* AD_43SNP.zip: SNP feature files
* reukbb.zip: Imaging feature files

Subjects must be aligned across SNP and imaging files.


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
```python
git clone https://github.com/QSong-github/AlzCLIP
```

Run inference with ensemble voting:
```python
python infer.py --model_path path_to_trained_model.pth --data_path path_to_processed_data --output_dir ./output
```

### Running Details
#### Training AlzCLIP
Run `main.py` to:
* Pretrain embeddings using contrastive loss
* Fine-tune embeddings for classification using cross-entropy loss


#### Ensemble Voting Inference
Run `infer.py` to:
* Extract SNP and MRI embeddings
* Train SVM, Random Forest, and XGBoost classifiers
* Perform soft voting (averaging predicted probabilities)
* Output prediction results and evaluation metrics

#### Tutorial Notebook
You can find detailed examples in the `tutorial notebook`, including:
* How to load pretrained AlzCLIP
* How to extract embeddings
* How to train ensemble classifiers (SVM, RF, XGB)
* How to ensemble vote and evaluate model performance

### Output
After inference, AlzCLIP generates:
* `predictions.csv`: containing true labels, predicted labels, and predicted probabilities.
* Model evaluation metrics: accuracy and AUC (Area Under Curve).
Outputs are saved under the specified --output_dir.

### Citation
If you use AlzCLIP in your research, please cite:
`Integrating Genetic and Imaging Data for Alzheimer’s Disease Diagnosis and Interpretation'


   
