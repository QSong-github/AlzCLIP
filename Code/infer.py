
import os
import torch
import numpy as np
import argparse
import pandas as pd
from model import UniModel
from dataset import SNPImageDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to trained AlzCLIP model')
parser.add_argument('--data_path', type=str, required=True, help='Path to processed dataset')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs')
args = parser.parse_args()

# 1. Load model
print("Loading model...")
model = UniModel()
model.load_state_dict(torch.load(args.model_path))
model.cuda()
model.eval()

# 2. Load dataset
print("Loading data...")
dataset = SNPImageDataset(args.data_path)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

all_img_embeds = []
all_snp_embeds = []
all_labels = []

# 3. Extract embeddings
print("Extracting embeddings...")
with torch.no_grad():
    for batch in dataloader:
        img, snp, label = batch
        img = img.cuda()
        snp = snp.cuda()
        label = label.cuda()

        img_embed = model.get_img_embeddings(img)
        snp_embed = model.get_snp_embeddings(snp)

        all_img_embeds.append(img_embed.cpu().numpy())
        all_snp_embeds.append(snp_embed.cpu().numpy())
        all_labels.append(label.cpu().numpy())

img_embeds = np.concatenate(all_img_embeds, axis=0)
snp_embeds = np.concatenate(all_snp_embeds, axis=0)
labels = np.concatenate(all_labels, axis=0)

# Concatenate embeddings
embeddings = np.concatenate([img_embeds, snp_embeds], axis=1)

# 4. Train Voting Classifiers
print("Training ensemble classifiers (SVM, RF, XGB)...")
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

svm = SVC(probability=True, random_state=42)
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

svm.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# 5. Soft Voting Prediction
print("Performing ensemble voting...")
svm_pred = svm.predict_proba(X_test)
rf_pred = rf.predict_proba(X_test)
xgb_pred = xgb.predict_proba(X_test)

avg_pred = (svm_pred + rf_pred + xgb_pred) / 3.0
y_pred = np.argmax(avg_pred, axis=1)

# 6. Evaluation
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, avg_pred[:, 1])

print(f"Ensemble Voting Accuracy: {acc:.4f}")
print(f"Ensemble Voting AUC: {auc:.4f}")

# 7. Save Results
os.makedirs(args.output_dir, exist_ok=True)
pd.DataFrame({
    'TrueLabel': y_test,
    'PredictedLabel': y_pred,
    'PredictedProb': avg_pred[:, 1]
}).to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)

print(f"Predictions saved to {args.output_dir}/predictions.csv")
