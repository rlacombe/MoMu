<<<<<<< HEAD
# MoMu

The PyTorch implementation of MoMu, described in "Natural Language-informed Understanding of Molecule Graphs".
=======

# GraphTextRetrieval

## 1. Checkpoints Preprocessing
You should download the folder from <https://huggingface.co/allenai/scibert_scivocab_uncased> (Pre-trained Sci-Bert), put it into the project and rename it as "bert_pretrained". Then you should put "MoMu-S" and "MoMu-K" into the folder "all_checkpoints".

## 2. Zeroshot Testing
#### zeroshot testing on our datasets with paragraph-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 0 --if_test 2 --if_zeroshot 1 --pth-test data/phy_data/test
```
#### zeroshot testing on our datasets with sentence-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 1 --if_test 2 --if_zeroshot 1 --pth-test data/phy_data/test
```
#### zeroshot testing on PCdes with paragraph-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 0 --if_test 2 --if_zeroshot 1 --pth-test data/kv_data/test
```
#### zeroshot testing on our datasets with sentence-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 1 --if_test 2 --if_zeroshot 1 --pth-test data/kv_data/test
```
## 3. Model Finetuning
#### finetuning on our datasets with paragraph-level and testing:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 0 --if_test 0 --if_zeroshot 0 --pth-test data/phy_data/test
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 0 --if_test 2 --if_zeroshot 0 --pth-test data/phy_data/test
```
#### finetuning on our datasets with sentence-level and testing:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 1 --if_test 0 --if_zeroshot 0 --pth-test data/phy_data/test
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 1 --if_test 2 --if_zeroshot 0 --pth-test data/phy_data/test
```
#### finetuning on PCdes with paragraph-level and testing:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 0 --if_test 0 --if_zeroshot 0 --pth-test data/kv_data/test
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 0 --if_test 2 --if_zeroshot 0 --pth-test data/kv_data/test
```
#### finetuning on PCdes with sentence-level and testing:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 1 --if_test 0 --if_zeroshot 0 --pth-test data/kv_data/test
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 1 --if_test 2 --if_zeroshot 0 --pth-test data/kv_data/test
```

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
  year={2022}
}
```
>>>>>>> 2c58e3e1 (add how to run)
