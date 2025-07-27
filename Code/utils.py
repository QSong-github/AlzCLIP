import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def Accuracy_score(pred, labels):
    """
    Calculate accuracy score
    Args:
        pred: predictions (can be logits, probabilities, or class indices)
        labels: true labels (class indices)
    """
    # Convert to numpy if tensors
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # If pred has multiple dimensions (probabilities/logits), get argmax
    if len(pred.shape) > 1:
        pred = np.argmax(pred, axis=1)
    
    # If labels has multiple dimensions, get argmax
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)
    
    return accuracy_score(labels, pred)


def F1_score(pred, labels, average='binary'):
    """Calculate F1 score"""
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    if len(pred.shape) > 1:
        pred = np.argmax(pred, axis=1)
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)
    
    return f1_score(labels, pred, average=average, zero_division=0)


def AUROC_score(pred, labels):
    """Calculate AUROC score"""
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # For AUROC, we need probabilities, not class predictions
    if len(pred.shape) > 1:
        # If we have probabilities, use the positive class probability
        if pred.shape[1] == 2:
            pred = pred[:, 1]  # Probability of positive class
        else:
            pred = np.max(pred, axis=1)  # Max probability for multi-class
    
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)
    
    try:
        return roc_auc_score(labels, pred)
    except ValueError:
        # If only one class present, return 0.5 (random performance)
        return 0.5


def Precision_score(pred, labels, average='binary'):
    """Calculate precision score"""
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    if len(pred.shape) > 1:
        pred = np.argmax(pred, axis=1)
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)
    
    return precision_score(labels, pred, average=average, zero_division=0)


def Recall_score(pred, labels, average='binary'):
    """Calculate recall score"""
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    if len(pred.shape) > 1:
        pred = np.argmax(pred, axis=1)
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)
    
    return recall_score(labels, pred, average=average, zero_division=0)


def evaluate_predictions(predictions, true_labels, probabilities=None):
    """
    Comprehensive evaluation of predictions
    
    Args:
        predictions: Predicted class labels
        true_labels: True class labels  
        probabilities: Predicted probabilities (optional, for AUROC)
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    results = {}
    
    results['accuracy'] = Accuracy_score(predictions, true_labels)
    results['f1'] = F1_score(predictions, true_labels)
    results['precision'] = Precision_score(predictions, true_labels)
    results['recall'] = Recall_score(predictions, true_labels)
    
    # Use probabilities for AUROC if available, otherwise use predictions
    auroc_input = probabilities if probabilities is not None else predictions
    results['auroc'] = AUROC_score(auroc_input, true_labels)
    
    return results


def print_evaluation_results(results, title="Evaluation Results"):
    """Print evaluation results in a formatted way"""
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"AUROC:     {results['auroc']:.4f}")


def test_evaluation_functions():
    """Test all evaluation functions"""
    print("Testing evaluation functions...")
    
    # Create test data
    true_labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9],
                             [0.9, 0.1], [0.6, 0.4], [0.7, 0.3], [0.2, 0.8]])
    
    try:
        # Test individual functions
        acc = Accuracy_score(predictions, true_labels)
        f1 = F1_score(predictions, true_labels)
        prec = Precision_score(predictions, true_labels)
        rec = Recall_score(predictions, true_labels)
        auroc = AUROC_score(probabilities, true_labels)
        
        print(f"✓ Accuracy: {acc:.4f}")
        print(f"✓ F1 Score: {f1:.4f}")
        print(f"✓ Precision: {prec:.4f}")
        print(f"✓ Recall: {rec:.4f}")
        print(f"✓ AUROC: {auroc:.4f}")
        
        # Test comprehensive evaluation
        results = evaluate_predictions(predictions, true_labels, probabilities)
        print_evaluation_results(results, "Comprehensive Test Results")
        
        print("\n✓ All evaluation function tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Evaluation function test failed: {e}")
        return False


if __name__ == "__main__":
    test_evaluation_functions()
