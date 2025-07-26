"""
AlzCLIP training function with two-stage training:
1. Pretrain embeddings using contrastive loss
2. Fine-tune embeddings for classification using cross-entropy loss
"""
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from dataset import CLIPDatasetUni
from model import CLIPModel_simple, ClassificationHead
from utils import AvgMeter, Accuracy_score
from torch.utils.data import DataLoader
import argparse


def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description='AlzCLIP')
    
    parser.add_argument('--save_path', type=str, default='./save', help='Save directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')   
    parser.add_argument('--pretrain_epochs', type=int, default=100, help='Pretraining epochs')
    parser.add_argument('--finetune_epochs', type=int, default=50, help='Fine-tuning epochs')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--pretrain_lr', type=float, default=0.001, help='Pretraining learning rate')
    parser.add_argument('--finetune_lr', type=float, default=0.0001, help='Fine-tuning learning rate')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--projection_dim', type=int, default=128, help='Projection dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for contrastive loss')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--dataset_path', type=str, default='./snp_lbl_img_dataset_ova', help='Dataset path')
    
    return parser.parse_args()


def setup_data(args):
    """Setup data loaders"""
    print("Setting up data loaders...")
    
    try:
        dataset = CLIPDatasetUni(args, dataset_path=args.dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating mock dataset for testing...")
        dataset = create_mock_dataset(args)
    
    size_all = len(dataset)
    train_size = int(0.8 * size_all)
    test_size = int(0.1 * size_all)
    infer_size = size_all - train_size - test_size
    
    train_dataset, test_dataset, infer_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size, infer_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    print(f'Train: {len(train_dataset)}, Test: {len(test_dataset)}, Infer: {len(infer_dataset)}')
    
    return train_loader, test_loader, infer_loader, train_dataset, test_dataset, infer_dataset


def create_mock_dataset(args):
    """Create a mock dataset for testing when real data is not available"""
    print("Creating mock dataset...")
    
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {
                'img': torch.randn(128),  # Mock MRI features
                'snp': torch.randn(234),  # Mock SNP features  
                'gt': torch.randint(0, 2, (1,)).item()  # Mock binary labels
            }
    
    return MockDataset()


def pretrain_epoch(model, train_loader, optimizer, args):
    """Pretraining with contrastive loss"""
    model.train()
    loss_meter = AvgMeter()
    
    for batch in tqdm(train_loader, desc="Pretraining"):
        # Move batch to device
        batch = {k: v.to(args.device) for k, v in batch.items()}
        
        # Forward pass for contrastive loss
        loss = model(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), len(batch['img']))
    
    return loss_meter


def finetune_epoch(model, classifier, train_loader, optimizer, criterion, args):
    """Fine-tuning with classification loss"""
    model.eval()  # Keep pretrained model frozen
    classifier.train()
    
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    
    for batch in tqdm(train_loader, desc="Fine-tuning"):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        
        # Get embeddings from frozen model
        with torch.no_grad():
            img_emb, snp_emb = model.get_embeddings(batch)
            
        # Combine embeddings
        combined_emb = torch.cat([img_emb, snp_emb], dim=1)
        
        # Classification
        logits = classifier(combined_emb)
        loss = criterion(logits, batch['gt'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, preds = torch.max(logits, 1)
        acc = (preds == batch['gt']).float().mean()
        
        loss_meter.update(loss.item(), len(batch['img']))
        acc_meter.update(acc.item(), len(batch['img']))
    
    return loss_meter, acc_meter


def validate_epoch(model, classifier, test_loader, criterion, args, stage="pretrain"):
    """Validation epoch"""
    model.eval()
    if classifier is not None:
        classifier.eval()
    
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Validation ({stage})"):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            if stage == "pretrain":
                loss = model(batch)
                loss_meter.update(loss.item(), len(batch['img']))
            else:
                img_emb, snp_emb = model.get_embeddings(batch)
                combined_emb = torch.cat([img_emb, snp_emb], dim=1)
                logits = classifier(combined_emb)
                loss = criterion(logits, batch['gt'])
                
                _, preds = torch.max(logits, 1)
                acc = (preds == batch['gt']).float().mean()
                
                loss_meter.update(loss.item(), len(batch['img']))
                acc_meter.update(acc.item(), len(batch['img']))
    
    return loss_meter, acc_meter


def main():
    # Get arguments
    args = get_args()
    print(f"Using device: {args.device}")
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Setup data
    train_loader, test_loader, infer_loader, train_dataset, test_dataset, infer_dataset = setup_data(args)
    
    # Save data splits for later use in inference
    torch.save({
        'train_dataset': train_dataset,
        'test_dataset': test_dataset, 
        'infer_dataset': infer_dataset,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'infer_size': len(infer_dataset),
        'args': args
    }, os.path.join(args.save_path, 'data_splits.pt'))
    
    # Initialize model
    model = CLIPModel_simple(args).to(args.device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # ================ Stage 1: Pretraining ================
    print("\n" + "="*50)
    print("Stage 1: Pretraining with Contrastive Loss")
    print("="*50)
    
    optimizer_pretrain = torch.optim.AdamW(
        model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay
    )
    
    best_pretrain_loss = float('inf')
    
    for epoch in range(args.pretrain_epochs):
        print(f"Pretrain Epoch: {epoch + 1}/{args.pretrain_epochs}")
        
        train_loss = pretrain_epoch(model, train_loader, optimizer_pretrain, args)
        val_loss, _ = validate_epoch(model, None, test_loader, None, args, stage="pretrain")
        
        print(f'Train Loss: {train_loss.avg:.4f}, Val Loss: {val_loss.avg:.4f}')
        
        if val_loss.avg < best_pretrain_loss:
            best_pretrain_loss = val_loss.avg
            torch.save(model.state_dict(), os.path.join(args.save_path, "pretrained_alzclip.pt"))
            print(f"✓ Best pretrained model saved! Loss: {best_pretrain_loss:.4f}")
    
    # ================ Stage 2: Fine-tuning ================
    print("\n" + "="*50)
    print("Stage 2: Fine-tuning for Classification") 
    print("="*50)
    
    # Load best pretrained model
    model.load_state_dict(torch.load(os.path.join(args.save_path, "pretrained_alzclip.pt")))
    
    # Freeze pretrained parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Add classification head
    classifier = ClassificationHead(
        input_dim=args.projection_dim * 2,
        hidden_dim=128,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(args.device)
    
    optimizer_finetune = torch.optim.AdamW(
        classifier.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    best_finetune_acc = 0.0
    
    for epoch in range(args.finetune_epochs):
        print(f"Finetune Epoch: {epoch + 1}/{args.finetune_epochs}")
        
        train_loss, train_acc = finetune_epoch(model, classifier, train_loader, optimizer_finetune, criterion, args)
        val_loss, val_acc = validate_epoch(model, classifier, test_loader, criterion, args, stage="finetune")
        
        print(f'Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc.avg:.4f}')
        print(f'Val Loss: {val_loss.avg:.4f}, Val Acc: {val_acc.avg:.4f}')
        
        if val_acc.avg > best_finetune_acc:
            best_finetune_acc = val_acc.avg
            torch.save({
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'args': args
            }, os.path.join(args.save_path, "finetuned_alzclip.pt"))
            print(f"✓ Best fine-tuned model saved! Acc: {best_finetune_acc:.4f}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Best pretraining loss: {best_pretrain_loss:.4f}")
    print(f"Best fine-tuning accuracy: {best_finetune_acc:.4f}")
    print(f"Models saved to: {args.save_path}")
    print("="*70)


if __name__ == "__main__":
    main()
