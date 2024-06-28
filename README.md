# BrainCLIP

This code is prepared for **"BrainClip"**.

## Overview

### Abstract


![The flowchart.](./model.png)

## Installation
Download BrainClip:
```git clone https://github.com/QSong-github/BrainCLIP```

Install Environment:
```pip install -r requirements.txt``` or ```conda env create -f environment.yml```


## Running

### Train the BrainClip with SNABLE.
   
   (1) Get the raw data.
   ```bash
   $ cd /path/to/AD_43SNP.zip
   $ unzip AD_43SNP.zip

   $ cd /path/to/reukbb.zip
   $ unzip reukbb.zip
   ```
   
   (2) Build the dataset.
   ```bash
   $ cd /path/to/Uni
   $ python dataset_making.py
   ```
   (3) Train the model.
   ```bash
   $ cd /path/to/Uni
   $ python main.py
   ```
   
   (4) Inference.
   ```bash
   $ cd /path/to/data
   $ python infer.py
   ```

   
### Inference

   We also provide partially processed data (2000 sequences) as demo, located under the ```./AntiFormer/subdt``` path. And the pre-trained model can be accessed from [google drive](https://drive.google.com/file/d/1D-mkFwoJzu7E__vJc3ahnFE4UVGYz4_Q/view?usp=sharing). Please download the model and put it into ```./AntiFormer/model_save``` directory.
   However，if you have processed all the data, you can replace the ```./subdt``` path with your data path for training by . And be careful to change the hyperparameters in the ```main.py``` to suit your hardware and target.


## Quick start

If you want to use our model, you can download the pre-trained BrainClip model from [here](https://github.com/QSong-github/BrainCLIP/tree/main/save) and quickly try it by the tutorial.

   
