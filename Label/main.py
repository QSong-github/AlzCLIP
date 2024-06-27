import os
from tqdm import tqdm
import torch
from torch import nn
import torch.utils.data.distributed
from dataset import CLIPDataset
from model import CLIPModel
from utils import AvgMeter
from torch.utils.data import DataLoader
import argparse

############################################## Initialization #########################################################
parser = argparse.ArgumentParser(description='DDP for CLIP')

parser.add_argument('--save_path', type=str, default='./save', help='')
parser.add_argument('--batch_size', type=int, default=128, help='')   # 072:128 0.0001
parser.add_argument('--max_epochs', type=int, default=200, help='')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--embedding_dim', type=int, default=256, help='')
parser.add_argument('--projection_dim', type=int, default=256, help='')
parser.add_argument('--dropout', type=float, default=0.1, help='')
parser.add_argument('--temperature', type=float, default=1.0, help='')
parser.add_argument('--vote', default=True, type=bool, help='')
parser.add_argument('--topk', type=int, default=3, help='')
parser.add_argument('--train_size_factor', type=float, default=0.8, help='')
parser.add_argument('--test_size_factor', type=float, default=0.1, help='')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='')
args = parser.parse_args()

print("Building loaders")
dataset = CLIPDataset(dataset_path='./lbl_img_dataset')

size_all = len(dataset)
train_size = int(0.8 * len(dataset))
test_size = int(0.1 * len(dataset))
infer_size = len(dataset) - train_size - test_size

assert size_all== train_size + infer_size + test_size

train_dataset, test_dataset, infer_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, infer_size], generator=torch.Generator().manual_seed(42))
print('train_dataset:', len(train_dataset), 'test_dataset:', len(test_dataset), 'infer_dataset:', len(infer_dataset))
print("dataset split completed")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)
infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)
print("Finished building loaders")

################################################ End of Initialization  ################################################

def train_epoch(model, train_loader, optimizer):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k == "img" or k == "lbl"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        count = batch["img"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg)

    return loss_meter


def test_epoch(model, test_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k == "img" or k == "lbl"}
        loss = model(batch)

        count = batch["img"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    print("Starting...")
    model = CLIPModel(args).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.max_epochs):
        print(f"Epoch: {epoch + 1}")

        model.train()
        train_loss = train_epoch(model, train_loader, optimizer)

        print('train_loss:', train_loss)

        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(model, test_loader)

        if test_loss.avg < best_loss:
            if not os.path.exists(str(args.save_path)):
                os.mkdir(str(args.save_path))
            best_loss = test_loss.avg
            best_epoch = epoch

            torch.save(model.state_dict(), str(args.save_path) + "/best.pt")
            print("Saved Best Model! Loss: {}".format(best_loss))

    print("Done!, final loss: {}".format(best_loss))
    print("Best epoch: {}".format(best_epoch))



if __name__ == "__main__":
    main()

