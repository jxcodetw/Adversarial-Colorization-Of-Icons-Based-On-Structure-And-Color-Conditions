# Adversarial Colorization Of Icons Based On Structure And Color Conditions
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)

Code for reproducing the experiments in the paper:
> Tsai-Ho Sun, Chien-Hsun Lai, Sai-Keung Wong, and Yu-Shuen Wang. 2019. Adversarial Colorization Of Icons Based On Structure And Color Conditions.
> In Proceedings of the 27th ACM International Conference on Multimedia (MM’19), October 21–25, 2019, Nice, France.
> ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3343031.3351041
> 
> **Abstract:** *We present a system to help designers create icons that are widely
used in banners, signboards, billboards, homepages, and mobile
apps. Designers are tasked with drawing contours, whereas our sys-
tem colorizes contours in different styles. This goal is achieved by
training a dual conditional generative adversarial network (GAN)
on our collected icon dataset. One condition requires the generated
image and the drawn contour to possess a similar contour, while
the other anticipates the image and the referenced icon to be similar
in color style. Accordingly, the generator takes a contour image
and a man-made icon image to colorize the contour, and then the
discriminators determine whether the result fulfills the two condi-
tions. The trained network is able to colorize icons demanded by
designers and greatly reduces their workload. For the evaluation,
we compared our dual conditional GAN to several state-of-the-art
techniques. Experiment results demonstrate that our network is
over the previous networks. Finally, we will provide the source
code, icon dataset, and trained network for public use.*

## Prerequisites
* pytorch
* torchvision
* tensorboardX

### Directory layout

    .
    ├── preprocessed_data
    │   ├── contour
    │   ├── img
    │   ├── labels.pt
    ├── build_dataset.py
    ├── dataset.py
    ├── dump_dataset.py
    ├── network.py
    ├── test.py
    └── README.md

## Usage
```
python train.py --dataset /path/to/dataset
```

## Dataset

preprocessed data: https://drive.google.com/file/d/1B8GjpBPpIxpAEy7cf8xpmA_ESy3CQXLh/view?usp=sharing
