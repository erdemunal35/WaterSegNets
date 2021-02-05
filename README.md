# WaterSegNets
 Semantic segmentation of water surfaces is a great challenge. This is due to the fact that the appearance of water is strongly changed by light influences, suspended solids and similar. Common segmentation methods are usually real-time capable only with powerful hardware. Within the scope of this project, encoder-decoder based semantic networks are trained and tested via [PyTorch](https://pytorch.org/).

##Requirements
[torch](https://pytorch.org/) ,[torchvision](https://pytorch.org/), [segmentation-models-pytorch](https://smp.readthedocs.io/en/latest/install.html) and [albumentations](https://pypi.org/project/albumentations/) libraries need to be installed.

[Netron](https://github.com/lutzroeder/Netron) and [mplcursors](https://pypi.org/project/mplcursors/) libraries might need to be installed later during testing of the networks.

## Dataset
 [Water Segmentation Dataset](https://www.kaggle.com/gvclsu/water-segmentation-dataset) is used for the train, valid and test dataset. You can download the dataset for free and resize all the images into 256x256 resolution by running [resizeImages.py](https://github.com/erdemunal35/WaterSegNets/blob/main/resizeImages.py) file. You can find the already resized [dataset_reshaped_256-256]() and [dataset_reshaped_480-320]() folders in this repository. However, it is recommended to run your own [resizeImages.py](https://github.com/erdemunal35/WaterSegNets/blob/main/resizeImages.py) file and create your own reshaped dataset since the reshaped dataset in this repo lack many example images because GitHub only allows 1000 images/files in one folder.
 
## Encoder Networks
> 1. [Very Deep Convolutional Networks (VGG)](https://arxiv.org/pdf/1409.1556.pdf)
> 2. [Residual Networks (ResNet)](https://arxiv.org/pdf/1512.03385.pdf)
> 3. [Densely Connected Convolutional Networks (DenseNet)](https://arxiv.org/pdf/1608.06993.pdf)
> 4. [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf)
> 5. [Squeeze-and-Excitation Networks (SE-Net)](https://arxiv.org/pdf/1709.01507.pdf)
> 6. [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)

## Decoder Networks
> 1. [LinkNet)](https://arxiv.org/pdf/1707.03718.pdf)
> 2. [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
> 3. [Feature Pyramid Networks (FPN)](https://arxiv.org/pdf/1612.03144.pdf)
> 4. [Pyramid Scene Parsing Network (PSPNet)](https://arxiv.org/pdf/1612.01105.pdf)
> 5. [Pyramid Attention Network (PAN)](https://arxiv.org/pdf/1805.10180.pdf)

## Train and Test
Check the files [train.py](https://github.com/erdemunal35/WaterSegNets/blob/main/train.py) and [test.py](https://github.com/erdemunal35/WaterSegNets/blob/main/test.py) comprehensively. You can train all the available [decoder networks](https://smp.readthedocs.io/en/latest/models.html) with [encoder networks](https://smp.readthedocs.io/en/latest/encoders.html) by changing some variable names in the train.py file. Then, you can test the trained models again by uncommenting such lines in the test.py file.
```
python train.py
```
```
python test.py
```
