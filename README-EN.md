# MMTV: Multimodal Model of Trans_EEGNet and Vision Transformer 

[![pCyV9AJ.png](https://s1.ax1x.com/2023/07/05/pCyV9AJ.png)](https://imgse.com/i/pCyV9AJ)



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12xHdlALR0JcB1-kgUzjXWo4X0WcZg-Po?usp=sharing)

[简体中文](./README.md)| English

## Introduction

MMTV is a model for diagnosing depression based on electroencephalogram (EEG) and tongue image information, taking into account both the patient's instantaneous brain state and long-term physical condition. The model consists of three modules: the EEG module, the tongue image module, and the multimodal module.

In the EEG module, we proposed a novel architecture, Trans_EEGNet, characterized by dual-stream input and a self-attention mechanism.

In the tongue image module, we designed a preprocessing and multi-step pre-training scheme for ViT, which combines image segmentation, meta-learning, and other methods.

In the multimodal module, we verified the correlation between the brain and the tongue, and used various methods for feature fusion.



## File/Directory Structure

It is recommended that readers only read `mmtv.ipynb`.

- `README.md`/`README-EN.md`: Provides basic information and usage instructions for the project.
- `overview.png`: An overview of the model.
- `mmtv.ipynb`: A demo of the model's overall operation.
- `EEGPreProcess/`: Preprocessing of EEG and multimodal data.
  - `DEofEEG.py`: Extracts the average differential entropy of each channel from .bdf files.
  - `EEGNetPre.py`: Processes the differential entropy of EEG into .npy files with dimensions such as channels and sequences, for use by architectures such as [EEGNet](https://github.com/aliasvishnu/EEGNet) and our Trans_EEGNet. In addition, image segmentation，rotating and cropping are performed on the tongue images here, which is convenient for use in multimodal situations.
  - `4DCRNNPre.py`: Processes the differential entropy of EEG into .npy files with dimensions such as two-dimensional brain maps and sequences, for use by architectures such as [4DCRNN](https://link.springer.com/article/10.1007/s11571-020-09634-1).
- `TongueLabelPreProcess/`: Processing of tongue image data labels.
  - `TongueLabelMap.py`: Reasonably converts the text labels of tongue images into numerical codes.
  - `EEG2TongueLabels.py`: Associates the labels of tongue images with the EEG segments of the same subject, facilitating the verification of brain-tongue correlation.

- `OnlineUtils/`: Online tools.
  - `sam.txt`: [SAM](https://segment-anything.com/) used for tongue image segmentation and its [derivative works](https://github.com/IDEA-Research/Grounded-Segment-Anything).
  - `paddletongue.txt`: Our open-source code (with dataset) for the single-modal tongue image model based on paddlepaddle.

- `OtherCodes/`: Completion code. As the directory is quite messy, it is not recommended to read. The core code has been detailed in `mmtv.ipynb`, and this part is only for supplementary understanding.
  - `keras_CRNN.ipynb`: The reproduced 4DCRNN architecture and its sub-modules and variant models, as well as the code for classification using traditional machine learning.
  - `Trans_EEGNet.ipynb`: Various variants of the Trans_EEGNet architecture, and demonstrations of how it handles other datasets.

