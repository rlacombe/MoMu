<<<<<<< HEAD
# MoMu

The PyTorch implementation of MoMu, described in "Natural Language-informed Understanding of Molecule Graphs".
=======

# GraphTextRetrieval
Source code for cross-modality retrieval for *[Natural Language-informed Understanding of Molecule Graphs](https://arxiv.org/abs/2209.05481)*. 
## Workspace Prepare
If you want to explore our job, you can following the instructions in this section
- Step 1: Download the zip or clone the repository to your workspace.
- Step 2: Download the `littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt` and `littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt` from [BaiduNetdisk](https://pan.baidu.com/share/init?surl=jvMP_ysQGTMd_2sTLUD45A)(the Password is 1234). Create a new directory by `mkdir all_checkpoints` and then put the downloaded model under the directory. Rename `littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt` to `MoMu-K.ckpt` and `littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt` to `MoMu-S.ckpt`
- Step 3: Download files from [Sci-Bert](https://huggingface.co/allenai/scibert_scivocab_uncased/tree/main). Create a new directory by `mkdir bert_pretrained` and then put these files under the directory.
- Step 4: Install python environment. Some important requirements are listed as follows(In fact, the environment is the almost same as [GraphTextPretrain](https://github.com/ddz16/GraphTextPretrain), so you do not need to install again if you have follow its instructions):
  ```
  Ubuntu 16.04.7
  python 3.8.13
  cuda 10.1

  # pytorch
  pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

  # torch_geometric 
  # you can download the following *.whl files in https://data.pyg.org/whl/
  wget https://data.pyg.org/whl/torch-1.8.0%2Bcu101/torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
  wget https://data.pyg.org/whl/torch-1.8.0%2Bcu101/torch_scatter-2.0.8-cp38-cp38-linux_x86_64.whl
  wget https://data.pyg.org/whl/torch-1.8.0%2Bcu101/torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl
  pip install torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
  pip install torch_scatter-2.0.8-cp38-cp38-linux_x86_64.whl
  pip install torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl
  pip install torch-geometric

  # transformers (4.18.0)
  pip install transformers 

  # rdkit
  pip install rdkit-pypi

  # ogb
  pip install ogb

  # pytorch_lightning (1.6.2)
  pip install pytorch_lightning 
## File Usage
The users may be going to use or edit the files below:
- main.py: Fine-tuning and testing code for cross-modality retrival. 
- data/
  - kv_data/: Pairs of (Graph, Text) data from  [KV-PLM](https://github.com/thunlp/KV-PLM) a.k.a PCdes
  - phy_data/: Pairs of (Graph, Text) data collected by us
- all_checkpoints/
  - MoMu-S.ckpt: Pretrained model  of MoMu-S
  - MoMu-K.ckpt: Pretrained model of MoMu-K
- data_provider/
  - match_dataset.py: Dataloader file
- model/
  - bert.py: Text encoder
  - gin_model.py: Graph encoder
  - constrastiv_gin.py Constrastive model with text encoder and graph encoder

## Zeroshot Testing
Zeroshot testing means cross-modality retrieval with origin MoMu. You can conduct zeroshot testing with differen settings as follows:
#### 1. zeroshot testing on phy_data with paragraph-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 0 --if_test 2 --if_zeroshot 1 --pth_test data/phy_data
```
#### 2. zeroshot testing on phy_data with sentence-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 1 --if_test 2 --if_zeroshot 1 --pth_test data/phy_data
```
#### 3. zeroshot testing on kv_data with paragraph-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 0 --if_test 2 --if_zeroshot 1 --pth_test data/kv_data/test
```
#### 4. zeroshot testing on kv_data with sentence-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 1 --if_test 2 --if_zeroshot 1 --pth_test data/kv_data/test
```
## Finetuning and Testing
To make MoMu satisfy the cross-modality retrieval task better, you can finetune MoMu and then test. Befor fintuning, you should create a new directory to save finetuned model by `mkdir finetune_save`. 
#### 1. finetuning on kv_data with paragraph-level and testing:
```
# finetune MoMu and save as 'finetune_save/finetune_para.pt '
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune_para.pt --data_type 0 --if_test 0 --if_zeroshot 0 

# test with fintuned model
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune_para.pt --data_type 0 --if_test 2 --if_zeroshot 0
```
#### 2. finetuning on kv_data with sentence-level and testing:
```
# finetune MoMu and save as 'finetune_save/finetune_sent.pt '
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune_sent.pt --data_type 1 --if_test 0 --if_zeroshot 0

# test with fintuned model
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune_sent.pt --data_type 1 --if_test 2 --if_zeroshot 0 
```
## Sample Result
Taking zeroshot testing on phy_data with paragraph-level as an example, we show the excuting result here.
It takes almost 10s to calculate the accuracy of retrieval, while calculating the Rec@20 takes about 2mins. 
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 0 --if_test 2 --if_zeroshot 1 --pth_test data/phy_data
Namespace(batch_size=64, data_type=0, epoch=30, graph_aug='dnodes', if_test=2, if_zeroshot=1, init_checkpoint='all_checkpoints/MoMu-S.ckpt', lr=5e-05, margin=0.2, output='finetune_save/sent_MoMu-S_73.pt', pth_dev='data/kv_data/dev', pth_test='data/phy_data', pth_train='data/kv_data/train', seed=73, text_max_len=128, total_steps=5000, warmup=0.2, weight_decay=0)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 87/87 [00:16<00:00,  5.31it/s]
Test Acc1: 0.4565587918015103
Test Acc2: 0.4317727436174038
Rec@20 1: 0.4579036317871269
Rec@20 2: 0.4348471772743617
```
<<<<<<< HEAD

<<<<<<< HEAD
## 4. Citation
Please refer to our paper:
>>>>>>> 2c58e3e1 (add how to run)

# Data availability

Our collected dataset consists of three folders holding molecular graphs, smiles and texts, respectively. The dataset can be downloaded in [https://pan.baidu.com/s/1hYbVOMhr7pwLkUtl5FWqbg](https://pan.baidu.com/s/1hYbVOMhr7pwLkUtl5FWqbg), the passward is **1234**. 
For cross-modality retrieval, the PCdes dataset is available at https://github.com/thunlp/KV-PLM. For text-to-molecule generation, the pre-trained MoFlow is available at https://github.com/calvin-zcx/moflow. For molecule caption, the ChEBI-20 dataset is directly available in our repository. For molecule property prediction, the eight datasets from MoleculeNet are available at https://github.com/deepchem/deepchem/tree/master/datasets and the processed datasets are available at: http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip. Please refer to the ReadMe in each part for the detailed data downloading and usage.

We will introduce our pretrain method and all the downstream tasks below. Please refer to the ReadMe in each part for the source code, system requirements, installation, demo, instructions for use, etc. 

# Pretrain

The pretrain code is available in the `Pretrain/` folder.

Our pretrained models MoMu-S and MoMu-K can be downloaded on [the Baidu Netdisk](https://pan.baidu.com/s/1jvMP_ysQGTMd_2sTLUD45A), the password is **1234**. All the downstream tasks use these two models.

```python
MoMu-K:   checkpoints/littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt
MoMu-S:   checkpoints/littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt
```

# Cross-modality retrieval

Since our MoMu model is pre-trained by matching weakly-correlated texts to corresponding molecular graphs, it is able to process both the graph and text modalities of molecules. We evaluate its performance in cross-modality retrieval. Given a molecule graph, graph-to-text (G-T) retrieval aims to retrieve the most relevant text descriptions of this molecule. Conversely, given a text paragraph, text-to-graph (T-G) retrieval aims at retrieving the most relevant molecule graph it describes. The code for these two downstream tasks is available in the `GraphTextRetrieval/` folder.

# Molecule caption

The molecule captioning task aims to generate texts to describe the given molecule. The code for this downstream task is available in the `MoleculeCaption/` folder.

# Zero-shot text-to-graph molecule generation

We propose a new task called zero-shot text-to-graph molecule generation. The goal is to design a cross modality molecule generator that takes as input the natural language description of the desired conditions and imagines new molecules that match the description. The code for this downstream task is available in the `Text2graph/` folder.

# Molecule property prediction

Molecular property prediction is a graph-level prediction task that is usually used to evaluate the transfer ability of pre-trained graph encoders. The code for this downstream task is available in the `MoleculePrediction/` folder.

# Citation

```
@article{su2022molecular,
<<<<<<< HEAD
  title={Natural Language-informed Understanding of Molecule Graphs},
  author={Bing Su, Dazhao Du, Zhao Yang, Yujie Zhou, Jiangmeng Li, Anyi Rao, Hao Sun, Zhiwu Lu, Ji-Rong Wen},
  year={2022}
}
```
=======
  title={A Molecular Multimodal Foundation **Model** Associating Molecule Graphs with Natural Language},
  author={Su, Bing and Du, Dazhao and Yang, Zhao and Zhou, Yujie and Li, Jiangmeng and Rao, Anyi and Sun, Hao and Lu, Zhiwu and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:2209.05481},
=======
=======
## Acknowledgment
This repository uses some code from [KV-PLM](https://github.com/thunlp/KV-PLM). Thanks to the original authors for their work!
>>>>>>> a401b3ce (modify)
## Citation
Please cite the following paper if you use the codes:

```
@article{su2022molecular,
  title={Natural Language-informed Understanding of Molecule Graphs},
  author={Bing Su, Dazhao Du, Zhao Yang, Yujie Zhou, Jiangmeng Li, Anyi Rao, Hao Sun, Zhiwu Lu, Ji-Rong Wen},
>>>>>>> bc5416f3 (modify)
  year={2022}
}
```
>>>>>>> 2c58e3e1 (add how to run)
