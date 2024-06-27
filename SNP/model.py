import torch
from torch import nn
import torch.nn.functional as F


class ImageEncoder_linear(nn.Module):
    def __init__(self,in_dim=128,out_dim=256):
        super().__init__()
        self.model = nn.Linear(in_features=in_dim,out_features=out_dim)

    def forward(self, x):
        return self.model(x)

class SNPEncoder_linear(nn.Module):
    def __init__(self,in_dim=234,out_dim=256):
        super().__init__()
        self.model = nn.Linear(in_features=in_dim,out_features=out_dim)

    def forward(self, x):
        return self.model(x)

class CLIPModel_simple(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.image_encoder = ImageEncoder_linear()
        self.snp_encoder = SNPEncoder_linear()
        self.image_projection = EmbedHead(self.args)
        self.snp_projection = EmbedHead(self.args)

    def forward(self, batch):
        image_features = self.image_encoder(batch['img'])
        snp_features = self.snp_encoder(batch['snp'])
        image_embeddings = self.image_projection(image_features)
        snp_embeddings = self.snp_projection(snp_features)

        # Calculating the Loss
        logits = (snp_embeddings @ image_embeddings.T) / self.args.temperature
        targets = F.softmax(
            (image_embeddings @ image_embeddings.T) / self.args.temperature, dim=-1
        )
        loss = cross_entropy(logits.T, targets.T, reduction='none')

        return loss.mean()



def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class EmbedHead(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.projection = nn.Linear(self.args.embedding_dim, self.args.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.args.projection_dim, self.args.projection_dim)
        self.dropout = nn.Dropout(self.args.dropout)
        self.layer_norm = nn.LayerNorm(self.args.projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x