# [CVPR2025] UniNet
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uninet-a-contrastive-learning-guided-unified/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=uninet-a-contrastive-learning-guided-unified) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uninet-a-contrastive-learning-guided-unified/anomaly-detection-on-btad)](https://paperswithcode.com/sota/anomaly-detection-on-btad?p=uninet-a-contrastive-learning-guided-unified)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uninet-a-contrastive-learning-guided-unified/anomaly-detection-on-mvtec-3d-ad-rgb)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-3d-ad-rgb?p=uninet-a-contrastive-learning-guided-unified)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uninet-a-contrastive-learning-guided-unified/anomaly-detection-on-visa)](https://paperswithcode.com/sota/anomaly-detection-on-visa?p=uninet-a-contrastive-learning-guided-unified)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uninet-a-contrastive-learning-guided-unified/image-classification-on-isic2018)](https://paperswithcode.com/sota/image-classification-on-isic2018?p=uninet-a-contrastive-learning-guided-unified)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uninet-a-contrastive-learning-guided-unified/anomaly-detection-on-ucsd-ped2)](https://paperswithcode.com/sota/anomaly-detection-on-ucsd-ped2?p=uninet-a-contrastive-learning-guided-unified)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uninet-a-contrastive-learning-guided-unified/medical-image-segmentation-on-cvc-colondb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-colondb?p=uninet-a-contrastive-learning-guided-unified)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uninet-a-contrastive-learning-guided-unified/medical-image-segmentation-on-cvc-clinicdb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-clinicdb?p=uninet-a-contrastive-learning-guided-unified)


![](figures/UniNet.jpg)

**UniNet: A Contrastive Learning-guided Unified Framework with Feature Selection for Anomaly Detection**

_Shun Wei, Jielin Jiang*, Xiaolong Xu_

[PDF Link](https://pangdatangtt.github.io/static/pdfs/UniNet__arXix_.pdf)

## 🔔 News
- 04-27-2025: The weight files have been uploaded.
- 04-26-2025: The code has been released (To be improved)!
- 04-21-2025: The code will be released in recent days!
- 02-27-2025: Accepted by CVPR2025!


## List of TODOs
- [x] 📖 Introduction
- [x] 🔧 Environments
- [ ] 📊 Data Preparation
- [ ] 🚀 Run Experiments
- [x] 📂 Results
- [x] 🔗 Citation
- [x] 🙏 Acknowledgements
- [x] 📜 License


## 📖 Introduction
This repository will contain source code for UniNet implemented with PyTorch.

UniNet is a unified framework designed for diverse domains, such as industrial, medical, and video, by addressing the limitations in general ability of existing methods.
Besides, UniNet can be suited for both unsupervised and supervised settings simultaneously. As a unified framework, UniNet also focuses on multi-class anomaly detection.

## 🔧 Environments
Create a new conda environment and install required packages.
```
conda create -n uninet_env python=3.9.7
conda activate uninet_env
pip install -r requirements.txt
```
Experiments are conducted on NVIDIA GeForce RTX 3090.
Same GPU and package version are recommended. 

## 📊 Data Preparation
The public datasets employed in the paper are listed below.
- Industrial domains: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/), [VisA](https://github.com/amazon-science/spot-diff/), [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip),
[MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad/downloads), [VAD](https://github.com/abc-125/vad?tab=readme-ov-file).
- Medical domains: [OCT2017](https://data.mendeley.com/datasets/rscbjbr9sj/3), [APTOS](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data), [ISIC2018](https://challenge.isic-archive.com/data/#2018), [Kvasir](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [CVC-ColonDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [CVC-ClinicDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579).
- Video domains: [Ped2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html).


MVTec AD
```
|-- mvtec
    |-- bottle
        |-- ground_truth
        |-- test
        |-- train
    |-- cable
        |-- ground_truth
        |-- test
        |-- train
    |-- ...
```
VisA
```
|-- VisA_pytorch
    |-- 1cls
        |-- candle
            |-- ground_truth
            |-- test
                    |-- good
                    |-- bad
            |-- train
                    |-- good
        |-- capsules
        |-- ....
```

## 📂 Results
### Unsupervised anomaly detection
![](figures/result1.jpg)
![](figures/result2.jpg)

### Supervised anomaly detection
![](figures/result3.jpg)

### Multi-class setting
![](figures/result4.jpg)

### Visualization results
<div align="center">
  <img src="figures/loc_results.jpg" width="377" style="display: inline-block;"/>
  <img src="figures/loc_results2.jpg" width="400" style="display: inline-block;"/>
</div>


## 🔗 Citation
Please cite our paper if the method and code help and inspire your project:

```bibtex
@inproceedings{wei2025uninet,
  title={UniNet: A Contrastive Learning-guided Unified Framework with Feature Selection for Anomaly Detection},
  author={Wei, Shun and Jiang, Jielin and Xu, Xiaolong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
}
```

## 🙏 Acknowledgements
We sincerely appreciate [ReContrast](https://github.com/guojiajeremy/ReContrast) for its concise and excellent approach.

## 📜 License
The code in this repository is licensed under the MIT license.
