# Multi-connection attention network

<div style="width:600px">
<img src="http://ftp.ipi.uni-hannover.de/ISPRS_WGIII_website/ISPRSIII_4_Test_results/2D_labeling_potsdam/top_resized_for_resultpage/top_mosaic_09cm_area2_13.tif_resized.jpg" alt="image" width="200" height="200">
  
<img src="http://ftp.ipi.uni-hannover.de/ISPRS_WGIII_website/ISPRSIII_4_Test_results/2D_labeling_potsdam/2D_labeling_Potsdam_details_SWJ_2/top_potsdam_2_13_class.tif_resized.jpg" alt="label" width="200" height="200">
  </div>

<!-- The pretrained model of Potsdam Dataset is available [here](https://drive.google.com/open?id=1jPS3MqWlqa1mwEvwhYxG9Gw8FTFTZHXH)-->

The pretrained model of Potsdam Dataset will be available in a month.

by Jicheng WANG, Li SHEN, Wenfan QIAO, Yanshuai DAI, Zhilin LI, details ([paper](https://www.mdpi.com/2072-4292/11/13/1617)).

## Introduction
This code is the implementation of SWJ_2 in [ISPRS Potsdam labeling challenge 2D](http://www2.isprs.org/commissions/comm2/wg4/potsdam-2d-semantic-labeling.html).
This network mainly consist of two module, i.e., multi-connection resnet and class-specific attention model.
## Installation
For installation, please follow the instructions of [tensorflow](tensorflow.org).Both GPU and CPU are compatible. Noted that the cuDNN is needed for GPU version. The version of tensorflow tested is 1.10.0.

## Usage
1. Clone this repository
```bash
git clone https://github.com/WindWang2/Multi-connection-attention-networks.git
```
2. Download the pretrained model and prepare the images
3. Change the code of test.py (path of pretrained model and directory of test images)
4. run the code
```bash
python3 test.py
```
train.py can be modified to train the model.
## Update
...
