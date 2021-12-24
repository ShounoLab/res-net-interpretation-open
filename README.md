# ResNet-interpretation

## Requierments
 
### Base

* Python 3.6.2
* CUDA 10.0
* cuDNN 7.6.5

#### Main modules

* pip
* PyTorch v1.1.0

* see [pip-requierments](pip_list.txt)

#### Local modules
* [Grad-CAM with Pytorch](https://github.com/gentaman/grad-cam-pytorch)
* [SVCCA](https://github.com/ShounoLab/svcca/tree/v2)

# How to setup

1. setup Python, CUDA, cuDNN, pip
1. download and install [Grad-CAM with Pytorch](https://github.com/gentaman/grad-cam-pytorch)
1. download and install [SVCCA](https://github.com/ShounoLab/svcca/tree/v2)

1. install some modules by using pip: 
```
$ pip install -r pip_list.txt
```


## Directories

* source codes. [src](src/)
* sandbox, test some thing. [notebooks](notebooks/)
* batch process. [batch\_process](batch_process/)


## Model

All models are trained on ImageNet

* ResNet-34 B
* PlainNet-34 B

## Method


* To compare ResNet vs. Plain ResNet
* To visualize the preferred stimulus in the receptive fields

## Papers or Technical Reports
* [A study of inner feature continuity of the ResNet model, JNNS2019](http://www.cns.pi.titech.ac.jp/JNNS2019/img/2019program.pdf)
* [Interpretation of ResNet by Visualization of Preferred Stimulus in Receptive Fields, PDPTA2020](https://arxiv.org/abs/2006.01645)
* [受容野の最適刺激を用いた畳込みニューラルネットワークの可視化手法, NC2020-47](https://www.ieice.org/ken/paper/20210303CCdq/)

