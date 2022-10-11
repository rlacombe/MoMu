<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# MoMu

The PyTorch implementation of MoMu, described in "Natural Language-informed Understanding of Molecule Graphs".
=======

# GraphTextRetrieval
Source code for cross-modality retrieval for *[Natural Language-informed Understanding of Molecule Graphs](https://arxiv.org/abs/2209.05481)*. 
Please go to [MoMu](https://github.com/ddz16/MoMu) to see the whole codebase.
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
=======
# MoleculeCaption
 This repository contains the code of the downstream task (molecule caption) in the paper "Natural Language-informed Understanding of Molecule Graphs”
<<<<<<< HEAD
>>>>>>> 44c5d02e (Initial commit)
=======

# Acknowledgment

We adapted the code of the PyTorch implementation of [MolT5](https://github.com/blender-nlp/MolT5/tree/main/evaluation). Thanks to the original authors for their work!

# Installation

The requirements for the evaluation code conda environment are in environment_eval.yml. An environment can be created using the following commands:

```
conda env create -n MolTextTranslationEval -f environment_eval.yml python=3.9
conda activate MolTextTranslationEval
python -m spacy download en_core_web_sm
pip install git+https://github.com/samoturk/mol2vec
```


# MolT5 & SciBert Checkpoints

Before jointly training the MolT5 model and GIN model in the molecule caption task, you should download three MolT5 checkpoints with different sizes on [huggingface](https://huggingface.co/laituan245). After downloading, put them into three folders `molt5-small/`, `molt5-base/`, `molt5-large/`, respectively. The final directory should look like this:

```
--molt5-small or molt5-base or molt5-large
  --config.json
  --pytorch_model.bin
  --special_tokens_map.json
  --spiece.model
  --tokenizer.json
  --tokenizer_config.json
```

Before evaluation, you should download the SciBert checkpoint on [huggingface](https://huggingface.co/allenai/scibert_scivocab_uncased). After downloading, put them into the `scibert/` folder. The final directory should look like this:

```
--scibert
  --config.json
  --flax_model.msgpack
  --pytorch_model.bin
  --vocab.txt
```

# Our Pretrained models

To better utilize the structural information of the input molecule for translation, we append the graph feature of the molecular graph to the inputs of the MolT5 encoder through a feature mapping module, which is implemented by a multi-layer perceptron. The graph features are extracted by the graph encoder GIN in MoMu-K or MoMu-S. You can download them on [the Baidu Netdisk](https://pan.baidu.com/s/1jvMP_ysQGTMd_2sTLUD45A), the password is **1234**. 

MoMu-K checkpoint:

```
checkpoints/littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt
```

MoMu-S checkpoint:

```
checkpoints/littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt
```

After downloading, you should put these two checkpoints into the `checkpoints/` folder.

# Finetune & Generate captions

Finetune on MoMu-S:
```
python main_transformer_smiles2caption.py --mode train --model_size base
```

or finetune on MoMu-K:
```
python main_transformer_smiles2caption.py --mode train --model_size base --MoMuK
```

After training process, you can generate the captions of all the molecules in the test dataset. Run these command:

```
python main_transformer_smiles2caption.py --mode test --model_size base --output_file out.txt
python main_transformer_smiles2caption.py --mode test --model_size base --output_file out.txt --MoMuK
```

A file `out.txt` will be generated. It contains all the SMILES strings, caption ground truths and captions generated by MoMu-enhanced MolT5, just like:
```
SMILES	ground truth	output
CCCCCCCCCCCCCCCCCCCCCC[C@H](C(=O)N[C@@H](CO[C@H]1[C@@H]([C@H]([C@H]([C@H](O1)CO)O)OS(=O)(=O)O)O)[C@@H](/C=C/CCCCCCCCCCCCC)O)O	The molecule is a galactosylceramide sulfate in which the sulfo group is located at position 3 and the ceramide N-acyl group is specified as (R)-2-hydroxylignoceroyl. It is a N-acyl-beta-D-galactosylsphingosine and a galactosylceramide sulfate. It derives from a (R)-2-hydroxylignoceric acid. It is a conjugate acid of a 1-(3-O-sulfo-beta-D-galactosyl)-N-[(2R)-2-hydroxylignoceroyl]sphingosine(1-).	The molecule is a galactosylceramide sulfate in which the ceramide N-acyl group is specified as (R)-2-hydroxybehenoyl. It is a galactosylceramide sulfate and a N-acyl-beta-D-galactosylsphingosine. It is a conjugate acid of a 1-(3-O-sulfo-beta-D-galactosyl)-N-[(2R)-2-hydroxybehenoyl]sphingosine(1-).
...
...
```


# Evaluation
To evaluate the molecule caption task, we just need the `out.txt` generated by MoMu-enhanced MolT5.
We provide an example `out.txt` in this repository. So you can evaluate the performance without running the finetune & generate captions commands.

## The BLEU & Rouge & Meteor metrics
Run this command to evaluate all NLG metrics:
```
python text_translation_metrics.py --input_file out.txt
```

## The Text2Mol metric
Before Evaluating the Text2Mol metric, download the `cid_to_smiles.pkl` file from https://uofi.box.com/v/MolT5-cid-to-smiles and `test_outputfinal_weights.320.pt` form https://uofi.box.com/s/es16alnhzfy1hpagf55fu48k49f8n29x. `cid_to_smiles.pkl` should be placed in the root path of the project and `test_outputfinal_weights.320.pt` should be placed in the `t2m_output/` folder. Then run this command:

```
python text_text2mol_metric.py --use_gt --input_file out.txt
```

# Citation

```
@article{su2022molecular,
  title={Natural Language-informed Understanding of Molecule Graphs},
  author={Bing Su, Dazhao Du, Zhao Yang, Yujie Zhou, Jiangmeng Li, Anyi Rao, Hao Sun, Zhiwu Lu, Ji-Rong Wen},
  year={2022}
}
```
>>>>>>> 45318f26 (init)
=======
# MoleculePrediction
 This repository contains the code of the downstream task (molecule property prediction) in the paper "Natural Language-informed Understanding of Molecule Graphs”
<<<<<<< HEAD
>>>>>>> 9ff1526d (Initial commit)
=======

# Acknowledgment

We adapted the code of the PyTorch implementation of [GraphCL](https://github.com/Shen-Lab/GraphCL/tree/master/transferLearning_MoleculeNet_PPI/chem). Thanks to the original authors for their work!

# Dependencies & Dataset

Please refer to https://github.com/snap-stanford/pretrain-gnns#installation for environment setup and https://github.com/snap-stanford/pretrain-gnns#dataset-download to download dataset.

If you cannot manage to install the old torch-geometric version, one alternative way is to use the new one (maybe ==1.6.0) and make some modifications based on this issue snap-stanford/pretrain-gnns#14. This might leads to some inconsistent results with those in the paper.

# Our Pretrained models

To apply MoMu, we use the graph encoder in the pre-trained MoMu-S and MoMu-K as the initialization, respectively. You can download them on [the Baidu Netdisk](https://pan.baidu.com/s/1jvMP_ysQGTMd_2sTLUD45A), the password is **1234**. We then fine-tune the graph encoder on the training sets of these datasets for predicting molecular properties, respectively.

MoMu-K checkpoint:

```
checkpoints/littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt
```

MoMu-S checkpoint:

```
checkpoints/littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt
```

After downloading, you should put these two checkpoints into the `checkpoints/` folder.

# Finetuning
Finetune on MoMu-K:

```
./finetune_MoMu-K.sh
```

Finetune on MoMu-S:

```
./finetune_MoMu-S.sh
```

Results will be recorded in `result.log`.

# Sample Result

Finetune MoMu-K on the muv dataset.

**Finetune process**:
```
MoleculeDataset(93087)
scaffold
Data(x=[15, 2], edge_attr=[30, 2], y=[17], edge_index=[2, 30], id=[1])

Iteration:   0%|          | 0/2328 [00:00<?, ?it/s]
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.001
    weight_decay: 0

Parameter Group 1
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.001
    weight_decay: 0
)
====epoch 1

Iteration:   0%|          | 1/2328 [00:00<14:27,  2.68it/s]
Iteration:   0%|          | 3/2328 [00:00<10:55,  3.55it/s]
Iteration:   0%|          | 6/2328 [00:00<08:18,  4.66it/s]
Iteration:   0%|          | 8/2328 [00:00<06:25,  6.02it/s]
Iteration:   0%|          | 10/2328 [00:00<05:08,  7.51it/s]
Iteration:   1%|          | 12/2328 [00:01<04:29,  8.59it/s]
Iteration:   1%|          | 14/2328 [00:01<04:01,  9.56it/s]
Iteration:   1%|          | 16/2328 [00:01<03:45, 10.23it/s]
Iteration:   1%|          | 18/2328 [00:01<03:32, 10.86it/s]
Iteration:   1%|          | 20/2328 [00:01<03:25, 11.21it/s]
Iteration:   1%|          | 22/2328 [00:01<03:20, 11.50it/s]
Iteration:   1%|          | 24/2328 [00:02<03:16, 11.73it/s]
Iteration:   1%|          | 26/2328 [00:02<03:07, 12.27it/s]
Iteration:   1%|          | 28/2328 [00:02<03:09, 12.15it/s]
Iteration:   1%|▏         | 30/2328 [00:02<03:14, 11.82it/s]
Iteration:   1%|▏         | 32/2328 [00:02<03:15, 11.73it/s]
Iteration:   1%|▏         | 34/2328 [00:02<03:17, 11.63it/s]
Iteration:   2%|▏         | 36/2328 [00:03<03:15, 11.73it/s]
Iteration:   2%|▏         | 38/2328 [00:03<03:10, 12.03it/s]
Iteration:   2%|▏         | 40/2328 [00:03<03:07, 12.19it/s]
Iteration:   2%|▏         | 42/2328 [00:03<03:12, 11.88it/s]
Iteration:   2%|▏         | 44/2328 [00:03<03:08, 12.09it/s]
Iteration:   2%|▏         | 46/2328 [00:03<03:02, 12.50it/s]
Iteration:   2%|▏         | 48/2328 [00:04<03:05, 12.28it/s]
Iteration:   2%|▏         | 50/2328 [00:04<02:54, 13.09it/s]
...
...
```

**Prediction results**:
```
muv 0 0.686400937903898
muv 1 0.7208213502256825
muv 2 0.6904045632349599
muv 3 0.6867226222897985
muv 4 0.746586288785911
muv 5 0.6702708041353815
muv 6 0.7474960366735247
muv 7 0.7077071846386775
muv 8 0.7278327257241599
muv 9 0.72664916270257
```
# Citation

```
@article{su2022molecular,
  title={Natural Language-informed Understanding of Molecule Graphs},
  author={Bing Su, Dazhao Du, Zhao Yang, Yujie Zhou, Jiangmeng Li, Anyi Rao, Hao Sun, Zhiwu Lu, Ji-Rong Wen},
  year={2022}
}
```
>>>>>>> 4241fe82 (init)
