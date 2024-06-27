"""
Read the data and make it into a dataset format for easy use.
"""
import csv
import os
from datasets import Dataset


# label_dict = {'AD': 0, 'CN': 1, 'EMCI': 2, 'LMCI':3, 'MCI':4, 'SMC':5, 'Patient':6}  # patient only 2 samples
label_dict = {'AD': 0, 'CN': 1, 'EMCI': 0, 'LMCI':0, 'MCI':0, 'SMC':0, 'Patient':0}  #




def raw2dataset_ova(snp_path,img_path):
    print('processing raw data...')
    # snp
    files = os.listdir(snp_path)
    print('SNP samples:',len(files))
    dict = {}
    for file in files:
        path = snp_path + file
        with open(path, 'r') as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            snp = []
            for row in csv_reader:
                snp.append(row[5]) # snp

            k = file[:-4]  # subject id
            v = snp
            dict[k] = v


    snp_all = []
    img_all = []
    lbl_all = []
    # img and label
    with open(img_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        cnt = 0
        cnt2 = 0
        for row in csv_reader:
            cnt2 = cnt2 + 1
            if row[1] in dict:
                group = row[2]
                img = row[12]
                snp = dict[row[1]]

                snp_all.append(snp)
                img_all.append(img)
                lbl_all.append(label_dict[group])
                cnt = cnt+1

    print('IMG samples:', cnt2)


    print('dataset samples',cnt)
    snp_img_dataset = {}
    snp_img_dataset['images'] = img_all
    snp_img_dataset['labels'] = lbl_all
    snp_img_dataset['snp'] = snp_all

    _dataset = Dataset.from_dict(snp_img_dataset)
    save_path = './snp_lbl_img_dataset_ova'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    _dataset.save_to_disk(save_path)

    print('dataset saved')


raw2dataset_ova('../AD_43SNP/AD_43SNP/','../output_merged.csv')