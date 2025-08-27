# ImageNet

## Prerequisites


```conda``` virtual environment is recommended.



```
conda create -n econvnext python=3.8 -y
```
Before install ohter requirements, you need to install [PaddlePaddle](https://www.paddlepaddle.org.cn/). We recommend using PaddlePaddle version 2.4.2 or above.
```
pip install -r requirements.txt
```

## Dataset

Download the ImageNet-1K dataset from [ImageNet official website](https://www.image-net.org/). The directory structure should be like:

```
  ├──dataset
  │   ├── imagenet
  │   │   ├── train
  │   │   │   ├── n01440764
  │   │   │   │   ├── n01440764_10026.JPEG
  │   │   ├── val
  │   │   │   ├── n01440764
  │   │   │   │   ├── ILSVRC2012_val_00000293.JPEG

```
## Training 

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
    -c ./ppcls/configs/ImageNet/ConvNext/E-ConvNext_tiny.yaml
```