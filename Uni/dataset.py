
import torch

from datasets import load_from_disk




class CLIPDatasetUni(torch.utils.data.Dataset):
    def __init__(self, args, dataset_path):
        self.path = dataset_path
        self.dataset = load_from_disk(self.path)


        self.imgs = self.dataset['images']
        self.snps = self.dataset['snp']
        self.lbls = self.dataset['labels']

        self.processed_img = self.imgs_processing(self.imgs)
        self.Uni_imgs = self.processed_img + self.processed_img
        self.lbls_, self.gt_ = self.lbls_processing(self.dataset['labels'])
        self.snps_ = self.snps_processing(self.dataset['snp'])
        self.Uni_lbls_snps = torch.cat((self.lbls_,self.snps_), dim=0)   # mix label and snp
        self.Uni_gt = torch.cat((self.gt_,self.gt_), dim=0)

        if args.train==True:
            # Label information can be included during training, so SNPs and labels are mixed.
            self.Uni_lbls_snps = torch.cat((self.lbls_, self.snps_), dim=0)
            self.Uni_gt = torch.cat((self.gt_, self.gt_), dim=0)
            self.Uni_imgs = self.processed_img + self.processed_img
        else:
            # Label information must not be included when testing or validating.
            self.Uni_lbls_snps = self.snps_
            self.Uni_gt = self.gt_
            self.Uni_imgs = self.processed_img


        print("Finished loading")

    def imgs_processing(self, image):
        img_list = []
        for i in image:
            i = i.strip("[]")
            i = i.split()

            i = [float(e) for e in i]
            img_list.append(torch.tensor(i))

        return img_list

    def lbls_processing(self, label):
        lbl_list = []
        for l in label:
            temp = []
            temp.append(l+3)  # token begins from 3. Token 0 1 2 for snp
            while len(temp)<234:
                temp.append(99)  # 99 as pad
            lbl_list.append(temp)
        return torch.tensor(lbl_list), torch.tensor(label)

    def snps_processing(self, snp):
        snp_list = []
        for s in snp:
            s_ = [float(e) if e != '' else 0 for e in s]
            if len(s_)>234:
                s_ = s_[:234]
            while len(s_)<234:
                s_.append(99)  # 99 as pad
            snp_list.append(s_)

        return torch.tensor(snp_list)


    def __getitem__(self, idx):
        item = {}

        item['img'] = self.Uni_imgs[idx]
        item['lbl_snp'] = self.Uni_lbls_snps[idx]
        item['gt'] =self.Uni_gt[idx]

        return item

    def __len__(self):
        return len(self.Uni_gt)


# dataset = CLIPDatasetUni(dataset_path='../snp_img_dataset_ovo')