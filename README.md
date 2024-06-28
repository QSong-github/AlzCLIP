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
   
### Inference   

   (4) Inference.
   ```bash
   $ cd /path/to/Uni
   $ python infer.py
   ```

### Train the BrainClip with SNP or Label.

We also provide training code using only SNPs or labels. If you want to try it, please visit the [SNP](https://github.com/QSong-github/BrainCLIP/tree/main/SNP) folder or [Label](https://github.com/QSong-github/BrainCLIP/tree/main/Label) folder.


## Quick start

If you want to use our model, you can download the pre-trained BrainClip model from [here](https://github.com/QSong-github/BrainCLIP/tree/main/save) and quickly try it by the [tutorial](https://github.com/QSong-github/BrainCLIP/blob/main/tutorial.ipynb).

   
