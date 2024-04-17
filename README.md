# CSFwinformer: Cross-Space-Frequency Window Transformer for Mirror Detection

This repo is the official implementation of ["CSFwinformer: Cross-Space-Frequency Window
Transformer for Mirror Detection (IEEE TIP 2024)"](https://ieeexplore.ieee.org/abstract/document/10462920).

## Installation

```
conda create -n md python=3.7 -y
conda activate md

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html

cd CSFwinformer

pip install -e .
pip install -r requirements/optional.txt

mkdir data

```

## Data Preparation

["MSD"](https://mhaiyang.github.io/ICCV2019_MirrorNet/index.html)

["PMD"](https://jiaying.link/cvpr2020-pgd/)

["RGBD-Mirror"](https://mhaiyang.github.io/CVPR2021_PDNet/index) 

You can download zip files for corresponding three datasets from ["here"](https://drive.google.com/drive/folders/1Fj0fIwn-mXI3xTlENiHXjYNLMUBRTZwg)

## Train
```
python tools/train.py configs/mirror/pmd_mirror_swin_small.py
```

## Test
```
python ./tools/test.py configs/mirror/pmd_mirror_swin_small.py work_dirs/pmd_mirror_swin_small/your_weight --show-dir ./results/pmd --eval mIoU
```

## Results and Models

| Dataset | Backbone| IoU↑ | Acc↑ | $F_β$↑ | MAE↓ | BER↓ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| PMD | swin_s | 69.84 | 77.28 | 0.849 | 0.024 | 11.91 |
| PMD | swin_b | 70.05 | 78.27 | 0.838 | 0.024 | 11.41 |
| MSD | swin_s | 82.13 | 88.72 | 0.895 | 0.046 | 7.15 |
| MSD | swin_b | 82.08 | 88.92 | 0.896 | 0.045 | 7.14 |
| RGBD-Mirror | swin_b | 78.66 | 84.64 | 0.900 | 0.031 | 8.57 |

You can find all weights from ["here"](https://drive.google.com/drive/folders/1f5NELOvgO0rH3n8IGyauruyoNYcWfO3J)

## Citation
If you find this repo useful for your research, please consider citing our paper:
```
@ARTICLE{10462920,
  author={Xie, Zhifeng and Wang, Sen and Yu, Qiucheng and Tan, Xin and Xie, Yuan},
  journal={IEEE Transactions on Image Processing}, 
  title={CSFwinformer: Cross-Space-Frequency Window Transformer for Mirror Detection}, 
  year={2024},
  volume={33},
  number={},
  pages={1853-1867},
  keywords={Mirrors;Feature extraction;Transformers;Frequency-domain analysis;Visualization;Semantics;Image segmentation;Mirror detection;texture analysis;cross-modality learning;frequency learning},
  doi={10.1109/TIP.2024.3372468}}
```