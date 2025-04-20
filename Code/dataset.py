
import torch

from datasets import load_from_disk

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.path = dataset_path
        self.dataset = load_from_disk(self.path)


        self.imgs = self.dataset['images']
        self.lbls = self.dataset['labels']
        self.snps = self.dataset['snp']

        print("Finished loading")

    def imgs_processing(self, image):
        image = image.strip("[]")
        image = image.split()

        image = [float(e) for e in image]

        return torch.tensor(image)

    def lbls_processing(self, label):
        return torch.tensor(float(label))

    def snps_processing(self, snp):
        snp = [float(e) if e != '' else 0 for e in snp]

        return torch.tensor(snp)

    def __getitem__(self, idx):
        item = {}

        image = self.imgs_processing(self.imgs[idx])
        snp = self.snps_processing(self.snps[idx])
        lbl = self.lbls_processing(self.lbls[idx])

        item['img'] = image
        item['lbl'] = lbl
        item['snp'] = snp

        return item

    def __len__(self):
        return len(self.lbls)


# dataset = CLIPDatasetUni(dataset_path='./snp_img_dataset_ovo')