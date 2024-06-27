"""
In the inference stage, you need to ensure that the model has been trained.
1. Enter the main function model_infer().
2. After calling the pre-trained model, get the vector representation of the sample through get_img_embeddings() and get_snp_embeddings().
3. Divide the data into training samples and verification samples.
4. Perform dot product similarity calculation through find_matches() to obtain the sample index that is most similar to the verification sample from the training sample.
5. Calculate the index. (voting or the highest probability of selection in this session)
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import CLIPModel_simple
import os
import numpy as np
from main import train_dataset, test_dataset, infer_dataset, train_size, test_size, infer_size, size_all
from main import args
from utils import Accuracy_score, F1_score, AUROC_score, Precision_score, Recall_score
dataset_all = torch.utils.data.ConcatDataset([train_dataset, test_dataset, infer_dataset])
assert len(dataset_all)== size_all
loader_all = DataLoader(dataset_all, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)

def get_img_embeddings(loader, model):
    model.eval()
    _image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader):
            image_features = model.image_encoder(batch["img"].cuda())
            image_embeddings = model.image_projection(image_features)
            _image_embeddings.append(image_embeddings)

    return torch.cat(_image_embeddings)


def get_snp_embeddings(loader, model):
    model.eval()
    _snabel_embeddings = []
    _lbl = []
    with torch.no_grad():
        for batch in tqdm(loader):
            _lbl.append(batch['gt'].cuda())
            snabel_features = model.snabel_encoder(batch["lbl_snp"].cuda())
            snabel_embeddings = model.snabel_projection(snabel_features)
            _snabel_embeddings.append(snabel_embeddings)

    return torch.cat(_snabel_embeddings), torch.cat(_lbl)


def find_matches(refer_embeddings, query_embeddings, topk):
    # find the closest matches
    refer_embeddings = torch.tensor(refer_embeddings).cuda()
    query_embeddings = torch.tensor(query_embeddings).cuda()
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    refer_embeddings = F.normalize(refer_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ refer_embeddings.T
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=topk)

    return indices.cpu().numpy()



def model_infer():
    print('Starting inferring.....')

    # load pretrained model
    model_path = args.save_path + "/best128.pt"
    print(model_path)
    model = CLIPModel_simple(args).to(args.device)
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    print("Finished loading model")

    # Get the vector representation of the image and Snable
    img_embeddings_all = get_img_embeddings(loader_all, model)
    snabel_embeddings_all, labels_all = get_snp_embeddings(loader_all, model)

    # What we get here is the vector representation of all data samples, including training samples.
    # Therefore, we need to distinguish between training samples and validation samples.
    img_embeddings_all = img_embeddings_all.cpu().numpy()
    snabel_embeddings_all = snabel_embeddings_all.cpu().numpy()
    labels_all = labels_all.cpu().numpy()
    print(img_embeddings_all.shape)
    print(snabel_embeddings_all.shape)
    print(labels_all.shape)
    print('all data processing finished')
    save_path = "./result/embeddings/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # The vector representation of the training sample is used as the reference,
    # and the vector of the validation sample is used as the query.
    refer_size = train_size + test_size
    query_size = infer_size

    # save refer
    img_embeddings = img_embeddings_all[:refer_size]
    snabel_embeddings = snabel_embeddings_all[:refer_size]
    labels = labels_all[:refer_size]

    print(img_embeddings.shape)
    print(snabel_embeddings.shape)
    print(labels.shape)
    np.save(save_path + "img_embeddings_refer.npy", img_embeddings.T)
    np.save(save_path + "snabel_embeddings_refer.npy", snabel_embeddings.T)
    np.save(save_path + "labels_refer.npy", labels)
    print('refer data saved')

    # save query
    img_embeddings = img_embeddings_all[refer_size:]
    snabel_embeddings = snabel_embeddings_all[refer_size:]
    labels = labels_all[refer_size:]

    print(img_embeddings.shape)
    print(snabel_embeddings.shape)
    print(labels.shape)
    np.save(save_path + "img_embeddings_query.npy", img_embeddings.T)
    np.save(save_path + "snabel_embeddings_query.npy", snabel_embeddings.T)
    np.save(save_path + "labels_query.npy", labels)
    print('query data saved')

    # using img embed to predict snp embed
    query = np.load(save_path + "img_embeddings_query.npy")
    refer = np.load(save_path + "img_embeddings_refer.npy")
    label_refer = np.load(save_path + "labels_refer.npy")
    label_query = np.load(save_path + "labels_query.npy")


    if query.shape[1]!=256:
        query = query.T
    if refer.shape[1]!=256:
        refer = refer.T
    # caculating accuracy
    # Retrieve the index of the most similar vector.
    index = find_matches(refer,query,topk=args.topk)
    if args.vote==True:
        # The voting mechanism compares the returned index samples
        # and their corresponding labels to make a vote.
        pred = []
        for i in range(len(index)):
            if np.sum(label_refer[index[i]] == 0)>np.sum(label_refer[index[i]] == 1):
                l = 0
            else:
                l = 1
            pred.append(l)
        true = label_query
        acc = Accuracy_score(pred, true)
        f1 = F1_score(pred, true)
        auroc = AUROC_score(pred, true)
        precision = Precision_score(pred, true)
        recall = Recall_score(pred, true)
        print('Accuracy:', acc, 'AUROC:', auroc,'Precision:', precision, 'Recall:', recall, 'F1:', f1)
    else:
        # Directly select the label of the index sample with the highest probability as the result.
        pred = label_refer[index[:,0]].reshape(-1)
        true = label_query
        acc = Accuracy_score(pred, true)
        f1 = F1_score(pred, true)
        auroc = AUROC_score(pred, true)
        precision = Precision_score(pred, true)
        recall = Recall_score(pred, true)
        print('Accuracy:', acc, 'AUROC:', auroc,'Precision:', precision, 'Recall:', recall, 'F1:', f1)




if __name__ == "__main__":
    model_infer()