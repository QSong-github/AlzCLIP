import torch
import torch.nn.functional as F
from pytz import reference
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import CLIPModel_simple
import os
import numpy as np
from main import train_dataset, test_dataset, infer_dataset, train_size, test_size, infer_size, size_all
from main import args

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
    _snp_embeddings = []
    _lbl = []
    with torch.no_grad():
        for batch in tqdm(loader):
            _lbl.append(batch["lbl"].cuda())
            snp_features = model.snp_encoder(batch['snp'].cuda())
            snp_embeddings = model.snp_projection(snp_features)
            _snp_embeddings.append(snp_embeddings)

    return torch.cat(_snp_embeddings), torch.cat(_lbl)

def find_matches(refer_embeddings, query_embeddings, topk):   # spot124 image3
    # find the closest matches
    refer_embeddings = torch.tensor(refer_embeddings).cuda()
    query_embeddings = torch.tensor(query_embeddings).cuda()
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    refer_embeddings = F.normalize(refer_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ refer_embeddings.T  # 76*2569
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=topk)

    return indices.cpu().numpy()

def model_infer():
    print('Starting inferring.....')
    model_path = args.save_path + "/best.pt"
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

    img_embeddings_all = get_img_embeddings(loader_all, model)
    snp_embeddings_all, labels_all = get_snp_embeddings(loader_all, model)

    img_embeddings_all = img_embeddings_all.cpu().numpy()
    snp_embeddings_all = snp_embeddings_all.cpu().numpy()
    labels_all = labels_all.cpu().numpy()
    print(img_embeddings_all.shape)
    print(snp_embeddings_all.shape)
    print(labels_all.shape)
    print('all data processing finished')
    save_path = "./result/embeddings/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    refer_size = train_size + test_size
    query_size = infer_size

    # save refer
    img_embeddings = img_embeddings_all[:refer_size]
    snp_embeddings = snp_embeddings_all[:refer_size]
    labels = labels_all[:refer_size]

    print(img_embeddings.shape)
    print(snp_embeddings.shape)
    print(labels.shape)
    np.save(save_path + "img_embeddings_refer.npy", img_embeddings.T)
    np.save(save_path + "snp_embeddings_refer.npy", snp_embeddings.T)
    np.save(save_path + "labels_refer.npy", labels)
    print('refer data saved')
    # save query
    img_embeddings = img_embeddings_all[refer_size:]
    snp_embeddings = snp_embeddings_all[refer_size:]
    labels = labels_all[refer_size:]
    print(img_embeddings.shape)
    print(snp_embeddings.shape)
    print(labels.shape)
    np.save(save_path + "img_embeddings_query.npy", img_embeddings.T)
    np.save(save_path + "snp_embeddings_query.npy", snp_embeddings.T)
    np.save(save_path + "labels_query.npy", labels)
    print('query data saved')

    query = np.load(save_path + "img_embeddings_query.npy")
    refer = np.load(save_path + "img_embeddings_refer.npy")
    label_refer = np.load(save_path + "labels_refer.npy")
    label_query = np.load(save_path + "labels_query.npy")



    if query.shape[1]!=256:
        query = query.T
    if refer.shape[1]!=256:
        refer = refer.T
    # caculating accuracy
    index = find_matches(refer,query,topk=args.topk)
    if args.vote==True:
        pred = []
        for i in range(len(index)):
            if np.sum(label_refer[index[i]] == 0)>np.sum(label_refer[index[i]] == 1):
                l = 0
            else:
                l =1
            pred.append(l)
        true = label_query
        correct_predictions = np.sum(pred == true)
        acc = correct_predictions / len(pred)
        print('Accuracy:',acc)
    else:
        pred = label_refer[index[:,0]].reshape(-1)
        true = label_query
        correct_predictions = np.sum(pred == true)
        acc = correct_predictions / len(pred)
        print('Accuracy:',acc)



if __name__ == "__main__":
    model_infer()