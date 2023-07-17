# A Model-Agnostic Framework for Universal Anomalym Detection of Multi-Organ and Multi-Modal Images — Pytorch Implementation

[comment]: <> (*The recent success of deep learning relies heavily on the large amount of labeled data. 
However, acquiring manually annotated symptomatic medical images is notoriously time-consuming and laborious, especially for rare or new diseases. In contrast, normal images from symptom-free healthy subjects without the need of manual annotation are much easier to acquire. In this regard, deep learning based anomaly detection approaches using only normal images are actively studied, achieving significantly better performance than conventional methods. Nevertheless, the previous works committed to develop a specific network for each organ and modality separately, ignoring the intrinsic similarity among images within medical field. In this paper, we propose a model-agnostic framework to detect the abnormalities of various organs and modalities with a single network. By imposing organ and modality classification constraints along with center constraint on the disentangled latent representation, the proposed framework not only improves the generalization ability of the network towards the simultaneous detection of anomalous images with various organs and modalities, but also boosts the performance on each single organ and modality. Extensive experiments with four different baseline models on three public datasets demonstrate the superiority of the proposed framework as well as the effectiveness of each component.*)

This is the official implementation of "A Model-Agnostic Framework for Universal Anomalym Detection of Multi-Organ and Multi-Modal Images". 
It includes experiments reported in the paper. The framework we propose can be applied to classic anomaly detection algorithms. Here we take our framework applied to [DPA](https://ieeexplore.ieee.org/abstract/document/9521238/) as an example to show the operation of the code. Thanks to the source code provided by the author of DPA. 

The structure of project will not be introduced too much here. For details, please refer to [the official implementation code](https://github.com/ninatu/anomaly_detection) introduction of DPA. We mainly introduce our framework code and how to apply our framework to DPA.


## MADDR
    anomaly_detection - python package; implementations of 
                                    MADDR_dpa
                                    └───rec_losses.py -- -- We redefine the L1Loss class function, by imposing organ and modality classification constraints along with center constraint on the disentangled latent representation.
     
## Installation 

Requirements: `Python3.6`
 
You can install miniconda environment(version 4.5.4):
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
bash Miniconda3-4.5.4-Linux-x86_64.sh
export PATH="{miniconda_root/bin}:$PATH
```

Installation:
```bash
pip install -r requirements.txt
pip install -e .
```

## Training and Evaluation 

The paper includes experiments on LiTS, COVID-CT and COVID-X-rays datasets. 

To reproduce all experiments of the paper, run:

```bash
python run_experiments.py
```
The training process and inference process are consistent with the baseline method, and we keep most parameters the same as the baseline methods, such as optimizer and learning rate. 

## Data Preprocessing 

1. Download data
(1)[COVID-CT-link](https://github.com/UCSD-AI4H/COVID-CT),
(2)[COVID-Xray-link](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen),
(3)[COVID-CT-link](https://competitions.codalab.org/competitions/17094)

2. Create a train split ("./folds/train_test_split/normal/train")

## Training models
You can download the original code of [DPA](https://github.com/ninatu/anomaly_detection), and then replace "rec_loss.py" file in the DPA code with "rec_loss.py" file we provide. Applying the MADDR framework to other baseline methods does the same. 

The official implementation of the baseline method:
1. [DPA](https://github.com/ninatu/anomaly_detection)
2. [f-AnoGAN](https://github.com/tSchlegl/f-AnoGAN)
3. [MemAE](https://github.com/donggong1/memae-anomaly-detection)
4. [GANomaly](https://github.com/samet-akcay/ganomaly)
