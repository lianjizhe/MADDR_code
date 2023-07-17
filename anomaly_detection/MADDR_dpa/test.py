import argparse
import yaml
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from anomaly_detection.dpa.train import Trainer
from anomaly_detection.utils.datasets import DatasetType, DATASETS
from anomaly_detection.utils.transforms import TRANSFORMS
from torch import nn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

def evaluate(config):
    batch_size = config['test_batch_size']
    results_root = config['results_root']
    model_path = config['test_model_path']

    os.makedirs(results_root, exist_ok=True)

    print(yaml.dump(config, default_flow_style=False))

    print("Starting model evaluation ...")

    enc, gen, image_rec_loss, (stage, resolution, progress, niter, _) = \
        Trainer.load_anomaly_detection_model(torch.load(model_path))
    enc, gen, image_rec_loss = enc.cuda().eval(), gen.cuda().eval(), image_rec_loss.cuda().eval()

    ########## train #########################
    dataset_type = config['train_dataset']['dataset_type']
    dataset_kwargs = config['train_dataset']['dataset_kwargs']
    transform_kwargs = config['train_dataset']['transform_kwargs']

    transform = TRANSFORMS[DatasetType[dataset_type]](**transform_kwargs)
    dataset = DATASETS[DatasetType[dataset_type]](
        transform=transform,
        **dataset_kwargs
    )

    # 保存了重建图片和原图
    # norm_anomaly_scores = predict_anomaly_scores(gen, enc, image_rec_loss, dataset, batch_size)
    # class_feature = get_class_preds(enc,dataset)

        ########## test_normal #########################
    dataset_type = config['test_datasets']['normal']['dataset_type']
    dataset_kwargs = config['test_datasets']['normal']['dataset_kwargs']
    transform_kwargs = config['test_datasets']['normal']['transform_kwargs']

    transform = TRANSFORMS[DatasetType[dataset_type]](**transform_kwargs)
    normal_dataset = DATASETS[DatasetType[dataset_type]](
        transform=transform,
        **dataset_kwargs
    )
    norm_anomaly_scores = predict_anomaly_scores(gen, enc, image_rec_loss, normal_dataset, batch_size, "Lits")
    ########## test_abnomaly #########################
    dataset_type = config['test_datasets']['anomaly']['dataset_type']
    dataset_kwargs = config['test_datasets']['anomaly']['dataset_kwargs']
    transform_kwargs = config['test_datasets']['anomaly']['transform_kwargs']

    transform = TRANSFORMS[DatasetType[dataset_type]](**transform_kwargs)
    anomaly_dataset = DATASETS[DatasetType[dataset_type]](
        transform=transform,
        **dataset_kwargs
    )

    norm_anomaly_scores = predict_anomaly_scores(gen, enc, image_rec_loss, anomaly_dataset, batch_size, "Lits")

# 对训练集的正常样本做分类
def get_class_preds(enc, dataset):
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=8)

    class_feature_list = []
    for images_labels,names in data_loader:
        with torch.no_grad():
            images = images_labels[0].cuda()
            labels = images_labels[1].cuda()
            feature_maps = enc(images)
            style_feature = feature_maps[:,:10,:,:]
            class_feature_list.append(style_feature)
    class_feature_cat = torch.cat(class_feature_list, dim = 0).cpu().squeeze(3).squeeze(2)
    # pdb.set_trace()

    return class_feature_cat

def predict_anomaly_scores(gen, enc, image_rec_loss, dataset, batch_size,image_path):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    data_loader = tqdm(data_loader)

    image_rec_loss.set_reduction('none')

    for images_labels,names in data_loader:
        images = images_labels
        # labels = images_labels[1]
        images = images.cuda()
        # labels = labels.cuda()
        with torch.no_grad():
            rec_images = gen(enc(images)).detach()

            # save images
            for i in range(len(images)):
                name = names[i].split('/')[-1]
                # pdb.set_trace()
                rec_image_np = rec_images[i].data.cpu().numpy()
                rec_image_np = (rec_image_np[0] + 1) / 2.0 * 255
                rec_image = Image.fromarray(rec_image_np).convert("L")
                rec_image = rec_image.resize((300, 300), Image.ANTIALIAS)
                # pdb.set_trace()
                rec_image.save("/data1/LJL/MICCAI/results/" + image_path + "/" + name[:-4] + "_rec" + name[-4:])

                image_np = images[i].data.cpu().numpy()
                image_np = (image_np[0] + 1) / 2.0 * 255
                image = Image.fromarray(image_np).convert("L")
                image = image.resize((300, 300), Image.ANTIALIAS)
                image.save("/data1/LJL/MICCAI/results/" + image_path + "/" + name)

    return 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('configs', type=str, nargs='*', help='Config paths')

    args = parser.parse_args()

    for config_path in args.configs:
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

        evaluate(config)


if __name__ == '__main__':
    main()

