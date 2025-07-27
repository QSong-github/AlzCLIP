import torch
from torch import nn
import torch.nn.functional as F


class ImageEncoder_linear(nn.Module):
    def __init__(self, in_dim=128, out_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=out_dim, out_features=out_dim)
        )

    def forward(self, x):
        return self.model(x)


class SNPEncoder_linear(nn.Module):
    def __init__(self, in_dim=234, out_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=out_dim, out_features=out_dim)
        )

    def forward(self, x):
        return self.model(x)


class ClassificationHead(nn.Module):
    """Classification head for fine-tuning"""
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class CLIPModel_simple(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_encoder = ImageEncoder_linear()
        self.snp_encoder = SNPEncoder_linear()
        self.image_projection = EmbedHead(self.args)
        self.snp_projection = EmbedHead(self.args)

    def forward(self, batch):
        """Forward pass for contrastive learning"""
        image_features = self.image_encoder(batch['img'])
        snp_features = self.snp_encoder(batch['snp'])
        image_embeddings = self.image_projection(image_features)
        snp_embeddings = self.snp_projection(snp_features)

        # Calculating the contrastive loss
        logits = (snp_embeddings @ image_embeddings.T) / self.args.temperature
        targets = F.softmax(
            (image_embeddings @ image_embeddings.T) / self.args.temperature, dim=-1
        )
        loss = cross_entropy(logits.T, targets.T, reduction='none')

        return loss.mean()
    
    def get_embeddings(self, batch):
        """Extract embeddings for inference - this was missing!"""
        image_features = self.image_encoder(batch['img'])
        snp_features = self.snp_encoder(batch['snp'])
        
        image_embeddings = self.image_projection(image_features)
        snp_embeddings = self.snp_projection(snp_features)
        
        return image_embeddings, snp_embeddings


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class EmbedHead(nn.Module):
    def __init__(self, args):
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


def test_model():
    """Test model functionality"""
    print("Testing model functionality...")
    
    class MockArgs:
        def __init__(self):
            self.embedding_dim = 256
            self.projection_dim = 128
            self.dropout = 0.1
            self.temperature = 1.0
    
    try:
        args = MockArgs()
        model = CLIPModel_simple(args)
        
        # Test batch
        batch = {
            'img': torch.randn(4, 128),
            'snp': torch.randn(4, 234),
            'gt': torch.randint(0, 2, (4,))
        }
        
        # Test contrastive forward
        loss = model(batch)
        print(f"✓ Contrastive loss: {loss.item():.4f}")
        
        # Test embedding extraction
        img_emb, snp_emb = model.get_embeddings(batch)
        print(f"✓ Image embeddings: {img_emb.shape}")
        print(f"✓ SNP embeddings: {snp_emb.shape}")
        
        # Test classification head
        classifier = ClassificationHead(input_dim=128*2, num_classes=2)
        combined = torch.cat([img_emb, snp_emb], dim=1)
        logits = classifier(combined)
        print(f"✓ Classification logits: {logits.shape}")
        
        print("✓ All model tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


if __name__ == "__main__":
    test_model()
