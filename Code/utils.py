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


from sklearn.metrics import accuracy_score
import torch


def Accuracy_score(pred, labels):
    max_prob_index_pred = torch.argmax(pred, dim=1).cpu()
    max_prob_index_labels = torch.argmax(labels, dim=1).cpu()
    acc = accuracy_score(max_prob_index_labels, max_prob_index_pred)

    return acc