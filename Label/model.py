import torch
from torch import nn
import torch.nn.functional as F



class ImageEncoder_linear(nn.Module):
    def __init__(self,in_dim=128,out_dim=256):
        super().__init__()
        self.model = nn.Linear(in_features=in_dim,out_features=out_dim)

    def forward(self, x):
        return self.model(x)

class LabelEncoder_linear(nn.Module):
    def __init__(self,in_dim=1,out_dim=256):
        super().__init__()
        self.model = nn.Linear(in_features=in_dim,out_features=out_dim)

    def forward(self, x):
        return self.model(x)

class CLIPModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.image_encoder = ImageEncoder_linear()
        self.lbl_encoder = LabelEncoder_linear()
        self.image_projection = EmbedHead(self.args)
        self.lbl_projection = EmbedHead(self.args)
        self.temperature = 1.0


    def forward(self, batch):
        image_features = self.image_encoder(batch['img'])
        lbl_features = self.lbl_encoder(batch['lbl'].unsqueeze(1))
        image_embeddings = self.image_projection(image_features)
        lbl_embeddings = self.lbl_projection(lbl_features)

        # Calculating the Loss
        logits = (lbl_embeddings @ image_embeddings.T) / self.temperature
        targets = F.softmax(
            (image_embeddings @ image_embeddings.T) / self.temperature, dim=-1
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
        self.projection = nn.Linear(args.embedding_dim, args.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(args.projection_dim, args.projection_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x