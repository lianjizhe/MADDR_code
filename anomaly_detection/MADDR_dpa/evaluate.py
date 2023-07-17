import argparse
import yaml
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import skimage.transform
import skimage.io
from torchvision.utils import make_grid
import sklearn
from anomaly_detection.dpa.train import Trainer
from anomaly_detection.utils.datasets import DatasetType, DATASETS
from anomaly_detection.utils.transforms import TRANSFORMS
from PIL import Image
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pdb

# #############################################################################################################


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

    dataset_type = config['test_datasets']['normal']['dataset_type']
    dataset_kwargs = config['test_datasets']['normal']['dataset_kwargs']
    transform_kwargs = config['test_datasets']['normal']['transform_kwargs']

    transform = TRANSFORMS[DatasetType[dataset_type]](**transform_kwargs)
    normal_dataset = DATASETS[DatasetType[dataset_type]](
        transform=transform,
        **dataset_kwargs
    )

    dataset_type = config['test_datasets']['anomaly']['dataset_type']
    dataset_kwargs = config['test_datasets']['anomaly']['dataset_kwargs']
    transform_kwargs = config['test_datasets']['anomaly']['transform_kwargs']
    transform = TRANSFORMS[DatasetType[dataset_type]](**transform_kwargs)
    anomaly_dataset = DATASETS[DatasetType[dataset_type]](
        transform=transform,
        **dataset_kwargs
    )

    norm_anomaly_scores = predict_anomaly_scores(gen, enc, image_rec_loss, normal_dataset, batch_size)
    an_anomaly_scores = predict_anomaly_scores(gen, enc, image_rec_loss, anomaly_dataset, batch_size)



    y_true = np.concatenate((np.zeros_like(norm_anomaly_scores), np.ones_like(an_anomaly_scores)))
    y_pred = np.concatenate((np.array(norm_anomaly_scores), np.array(an_anomaly_scores)))


    labels = list(y_true)
    preds = list(y_pred)
    with open("result_lits_no.txt","w",encoding="utf-8") as fw:
        for i in range(len(labels)):
            fw.write(str(labels[i]) + '\t' + str(preds[i]) + '\n')

    # -----------------------------------------------------------------------------------
    # 保存预测的loss值和真是标签
    # label = list(y_true)
    # pred = list(y_pred)
    # with open("result.txt","w",encoding="utf-8") as fw:
    #     for i in range(len(label)):
    #         fw.write(str(label[i]) + '\t' + str(pred[i]) + '\n')


    # -------------------------------------------------------------------------------------
    # 对预测的值进行归一化
    tmp_min = min(y_pred)
    tmp_max = max(y_pred)
    y_pred = (y_pred - tmp_min) / (tmp_max - tmp_min)

    roc_auc = roc_auc_score(y_true, y_pred)
    results = pd.DataFrame([[niter, roc_auc]], columns=['niter', 'ROC AUC'])
    print("Model evaluation is complete. Results: ")
    print(results)

    label = list(y_true)
    pred = list(y_pred)
    with open("result_lits.txt","w",encoding="utf-8") as fw:
        for i in range(len(label)):
            fw.write(str(label[i]) + '\t' + str(pred[i]) + '\n')


    # --------------------------------------------------------------------------------------
    # 取不同的阈值计算精度召回率
    y_true = np.concatenate((np.zeros_like(norm_anomaly_scores), np.ones_like(an_anomaly_scores)))
    y_pred = np.concatenate((np.array(norm_anomaly_scores), np.array(an_anomaly_scores)))
    tmp_min = min(y_pred)
    tmp_max = max(y_pred) 
    y_pred = (y_pred - tmp_min) / (tmp_max - tmp_min)
    y_true_list = list(y_true)
    # 将ssim_pred中预测的值按照阈值置为0或1
    step = (tmp_max - tmp_min) / 1000.0
    thresholds = list(np.arange(tmp_min, tmp_max, step))
    max_acc = 0
    y_preds_max = []
    best_threshold = 0 
    for threshold in thresholds:
        y_preds = []
        y_pred_list = list(y_pred)
        for i in range(len(y_pred_list)):
            if y_pred[i] < threshold:
                y_preds.append(0.0)
            else:
                y_preds.append(1.0)
        acc = metrics.accuracy_score(y_true_list, y_preds, normalize=True, sample_weight=None)
        if acc > max_acc:
            max_acc = acc
            y_preds_max = y_preds
            best_threshold = threshold
    print(metrics.classification_report(y_true, y_preds_max))
    print(confusion_matrix(y_true, y_preds_max))
    print("threshold:",best_threshold)



def predict_anomaly_scores(gen, enc, image_rec_loss, dataset, batch_size):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    data_loader = tqdm(data_loader)

    image_rec_loss.set_reduction('none')

    anomaly_scores = []
    for images,names in data_loader:
        images = images.cuda()

        with torch.no_grad():
            rec_images = gen(enc(images)).detach()
            cur_anomaly_scores = image_rec_loss(images, rec_images)

            # save images
            for i in range(len(images)):
                name = names[i].split('/')[-1]
                # pdb.set_trace()
                rec_image_np = rec_images[i].data.cpu().numpy()
                rec_image_np = (rec_image_np[0] + 1) / 2.0 * 255
                rec_image = Image.fromarray(rec_image_np).convert("L")
                rec_image = rec_image.resize((300, 300), Image.ANTIALIAS)
                scores_str = cur_anomaly_scores[i].item()
                # pdb.set_trace()
                rec_image.save("/data1/LJL/MICCAI/results/output_data/LiTs/rec/" + name[:-4]+"_"+str(scores_str)[:5] + "_rec" + name[-4:])

                image_np = images[i].data.cpu().numpy()
                image_np = (image_np[0] + 1) / 2.0 * 255
                image = Image.fromarray(image_np).convert("L")
                image = image.resize((300, 300), Image.ANTIALIAS)
                image.save("/data1/LJL/MICCAI/results/output_data/LiTs/orign/" + name)
                
        anomaly_scores.extend(cur_anomaly_scores.detach().cpu().numpy())

    return anomaly_scores


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











# import argparse
# import yaml
# import os
# import torch
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from sklearn.metrics import roc_auc_score
# import numpy as np
# import pandas as pd
# # import skimage.transform
# # import skimage.io
# # from torchvision.utils import make_grid
# # import sklearn

# import sys
# sys.path.remove(sys.path[0])
# sys.path.append("/home/jiaoliang/MICCAI/yhz_anomaly_detection")

# from anomaly_detection.dpa.train import Trainer
# from anomaly_detection.utils.datasets import DatasetType, DATASETS
# from anomaly_detection.utils.transforms import TRANSFORMS


# def evaluate(config):
#     batch_size = config['test_batch_size']
#     results_root = config['results_root']
#     model_path = config['test_model_path']

#     os.makedirs(results_root, exist_ok=True)

#     print(yaml.dump(config, default_flow_style=False))

#     print("Starting model evaluation ...")

#     enc, gen, image_rec_loss, (stage, resolution, progress, niter, _) = \
#         Trainer.load_anomaly_detection_model(torch.load(model_path))
#     enc, gen, image_rec_loss = enc.cuda().eval(), gen.cuda().eval(), image_rec_loss.cuda().eval()

#     dataset_type = config['test_datasets']['normal']['dataset_type']
#     dataset_kwargs = config['test_datasets']['normal']['dataset_kwargs']
#     transform_kwargs = config['test_datasets']['normal']['transform_kwargs']

#     transform = TRANSFORMS[DatasetType[dataset_type]](**transform_kwargs)
#     normal_dataset = DATASETS[DatasetType[dataset_type]](
#         transform=transform,
#         **dataset_kwargs
#     )

#     dataset_type = config['test_datasets']['anomaly']['dataset_type']
#     dataset_kwargs = config['test_datasets']['anomaly']['dataset_kwargs']
#     transform_kwargs = config['test_datasets']['anomaly']['transform_kwargs']
#     transform = TRANSFORMS[DatasetType[dataset_type]](**transform_kwargs)
#     anomaly_dataset = DATASETS[DatasetType[dataset_type]](
#         transform=transform,
#         **dataset_kwargs
#     )

#     norm_anomaly_scores = predict_anomaly_scores(gen, enc, image_rec_loss, normal_dataset, batch_size)
#     an_anomaly_scores = predict_anomaly_scores(gen, enc, image_rec_loss, anomaly_dataset, batch_size)

#     # # visualization
#     # norm_images, norm_rec_images, norm_maps, norm_anomaly_scores = \
#     #     predict_all(gen, enc, image_rec_loss, normal_dataset, batch_size)
#     # an_images, an_rec_images, an_maps, an_anomaly_scores = \
#     #     predict_all(gen, enc, image_rec_loss, anomaly_dataset, batch_size)
#     #
#     # # Process masks for better visualization
#     # norm_maps = np.array(norm_maps)
#     # an_maps = np.array(an_maps)
#     # map_values = np.concatenate((norm_maps.flatten(), an_maps.flatten()))
#     #
#     # # Delete zero for better visualization
#     # map_values = map_values[map_values != 0]
#     # maps_min, maps_max = np.quantile(map_values, 0.05), np.quantile(map_values, 0.95)
#     #
#     # norm_maps = (np.clip(norm_maps, maps_min, maps_max) - maps_min) / (maps_max - maps_min)
#     # an_maps = (np.clip(an_maps, maps_min, maps_max) - maps_min) / (maps_max - maps_min)
#     #
#     # # Save normal
#     # output_dir = os.path.join(results_root, 'normal')
#     # os.makedirs(output_dir, exist_ok=True)
#     # visualize(norm_images, norm_rec_images, norm_maps, norm_anomaly_scores, output_dir)
#     # df_anomaly_scores = pd.DataFrame(norm_anomaly_scores[:, np.newaxis], columns=['Anomaly Scores'])
#     # df_anomaly_scores.to_csv(os.path.join(output_dir, 'anomaly_scores.csv'))
#     #
#     # # Save anomaly
#     # output_dir = os.path.join(results_root, 'anomaly')
#     # os.makedirs(output_dir, exist_ok=True)
#     # visualize(an_images, an_rec_images, an_maps, an_anomaly_scores, output_dir)
#     # df_anomaly_scores = pd.DataFrame(an_anomaly_scores[:, np.newaxis], columns=['Anomaly Scores'])
#     # df_anomaly_scores.to_csv(os.path.join(output_dir, 'anomaly_scores.csv'))

#     y_true = np.concatenate((np.zeros_like(norm_anomaly_scores), np.ones_like(an_anomaly_scores)))
#     y_pred = np.concatenate((np.array(norm_anomaly_scores), np.array(an_anomaly_scores)))

#     roc_auc = roc_auc_score(y_true, y_pred)

#     output_path = os.path.join(results_root, 'results.csv')

#     results = pd.DataFrame([[niter, roc_auc]], columns=['niter', 'ROC AUC'])
#     print("Model evaluation is complete. Results: ")
#     print(results)

#     results.to_csv(output_path, index=False)


# def predict_anomaly_scores(gen, enc, image_rec_loss, dataset, batch_size):
#     data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)
#     data_loader = tqdm(data_loader)

#     image_rec_loss.set_reduction('none')

#     anomaly_scores = []
#     for images in data_loader:
#         images = images.cuda()
#         with torch.no_grad():
#             rec_images = gen(enc(images)).detach()
#             cur_anomaly_scores = image_rec_loss(images, rec_images)
#         anomaly_scores.extend(cur_anomaly_scores.detach().cpu().numpy())

#     return anomaly_scores


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--visualize', action="store_true")
#     parser.add_argument('configs', type=str, nargs='*', help='Config paths')

#     args = parser.parse_args()

#     for config_path in args.configs:
#         with open(config_path, 'r') as stream:
#             config = yaml.load(stream, Loader=yaml.FullLoader)

#         evaluate(config)


# if __name__ == '__main__':
#     main()


# # def predict_all(gen, enc, image_rec_loss, dataset, batch_size):
# #     data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)
# #     data_loader = tqdm(data_loader)
# #
# #     save_images = []
# #     save_rec_images = []
# #     maps = []
# #     anomaly_scores = []
# #
# #     torch.set_grad_enabled(True)
# #
# #     for images in data_loader:
# #         images = images.cuda()
# #         rec_images = gen(enc(images)).detach()
# #
# #         save_images.extend(images.detach().cpu().numpy())
# #         save_rec_images.extend(rec_images.detach().cpu().numpy())
# #
# #         image_rec_loss.set_reduction('none')
# #         anomaly_scores.extend(image_rec_loss(images, rec_images).detach().cpu().numpy())
# #
# #         image_rec_loss.set_reduction('pixelwise')
# #         maps.extend(image_rec_loss(images, rec_images).detach().cpu().numpy())
# #
# #     return np.array(save_images), np.array(save_rec_images), np.array(maps), np.array(anomaly_scores)
# #
# #
# # def visualize(images, recs, maps, anomaly_scores, output_dir):
# #     if images.shape[1] == 1:
# #         images = np.concatenate((images, images, images), axis=1)
# #         recs = np.concatenate((recs, recs, recs), axis=1)
# #
# #     for sort in ['fix', 'most_abnormal', 'less_abnormal', 'random']:
# #         if sort == 'random':
# #             images, recs, anomaly_scores, maps = sklearn.utils.shuffle(images, recs, anomaly_scores, maps)
# #         elif sort == 'fix':
# #             np.random.seed(31415)
# #             images, recs, anomaly_scores, maps = sklearn.utils.shuffle(images, recs, anomaly_scores, maps)
# #         else:
# #             order = np.argsort(anomaly_scores)
# #             if sort == 'most_abnormal':
# #                 order = order[::-1]
# #
# #             images = images[order]
# #             recs = recs[order]
# #             maps = maps[order]
# #             anomaly_scores = anomaly_scores[order]
# #
# #         nrow = 1
# #         ncol = 20
# #
# #         poster = []
# #         scores = []
# #         for i in range(nrow):
# #             cur_images = images[i * ncol: (i + 1) * ncol] * 0.5 + 0.5
# #             poster.extend(cur_images)
# #
# #             cur_recs = recs[i * ncol: (i + 1) * ncol] * 0.5 + 0.5
# #             poster.extend(cur_recs)
# #
# #             cur_maps = np.array(maps[i * ncol: (i + 1) * ncol])
# #             cur_maps = np.repeat(cur_maps, 3, axis=1)
# #             poster.extend(cur_maps)
# #
# #             gray_images = 0.2125 * cur_images[:, 0] + 0.7154 * cur_images[:, 1] + 0.0721 * cur_images[:, 2]
# #             gray_images = gray_images[:, np.newaxis]
# #             cur_maps = cur_maps[:, :1]
# #             red_maps = np.concatenate((cur_maps, 1 - cur_maps, np.zeros(cur_maps.shape)),  axis=1)
# #             overlay = 0.7 * gray_images + 0.3 * red_maps
# #
# #             poster.extend(overlay)
# #             scores.extend(anomaly_scores[i * ncol: (i + 1) * ncol])
# #
# #         poster = torch.from_numpy(np.array(poster))
# #         poster = make_grid(poster, nrow=ncol, padding=0).cpu().numpy().transpose(1, 2, 0)
# #         poster = skimage.img_as_ubyte(poster.clip(0, 1))
# #         skimage.io.imsave(os.path.join(output_dir, sort + '.png'), poster)
# #         with open(os.path.join(output_dir, sort + '.txt'), 'w') as fout:
# #             fout.write(' '.join(map(lambda x: f'{x:.2f}', scores)))
