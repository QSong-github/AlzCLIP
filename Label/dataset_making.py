import csv
import os
from datasets import Dataset


# label_dict = {'AD': 0, 'CN': 1, 'EMCI': 2, 'LMCI':3, 'MCI':4, 'SMC':5, 'Patient':6}  # patient only 2 samples
label_dict = {'AD': 0, 'CN': 1, 'EMCI': 0, 'LMCI':0, 'MCI':0, 'SMC':0, 'Patient':0}  #



def raw2dataset_ovo_img(snp_path,img_path):
    print('processing raw data...')
    # img and label
    with open(img_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        img_all = []
        lbl_all = []
        cnt = 0
        for row in csv_reader:
            group = row[2]
            img = row[12]
            lbl_all.append(label_dict[group])
            img_all.append(img)
            cnt = cnt+1
    print('IMG samples:',cnt)


    print('dataset samples',cnt)
    snp_img_dataset = {}
    snp_img_dataset['images'] = img_all
    snp_img_dataset['labels'] = lbl_all
    _dataset = Dataset.from_dict(snp_img_dataset)
    save_path = './lbl_img_dataset'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    _dataset.save_to_disk(save_path)

    print('dataset saved')




raw2dataset_ovo_img('../AD_43SNP/AD_43SNP/','../output_merged.csv')