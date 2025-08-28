# ImageNet

## Prerequisites

A ```conda``` virtual environment is recommended.

```
conda create -n econvnext python=3.8 -y
```
Before installing other requirements, you need to install [PaddlePaddle](https://www.paddlepaddle.org.cn/). We recommend using PaddlePaddle version 2.4.2 or above.
```
pip install -r requirements.txt
```

## Dataset

[Sonor Dataset](https://github.com/violetweir/Sonor_dataset)

[DUO Dataset](https://github.com/chongweiliu/DUO)

Please download and extract the datasets into the `dataset/` directory, and rename the dataset folders to `sonor` and `duo`.
The dataset directory structure should be as follows:

```
dataset
├── duo
│   ├── train
│   ├── val
│   └── test
└── sonor
  ├── train
  ├── val
  └── test
```
## Training

### How to train PP-YOLOE and YOLOv10

Before training PP-YOLOE and YOLOv10, please configure the data files `configs/datasets/DUO_coco.yml` and `configs/datasets/sonor_coco.yml` in the dataset directory.
A sample configuration is shown below:

```yaml
metric: COCO
num_classes: 4

TrainDataset:
  name: COCODataSet
  image_dir: DUO_pic/train
  anno_path: DUO_ANN/instances_train.json
  dataset_dir: dataset/DUO
  data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  name: COCODataSet
  image_dir: DUO_pic/test
  anno_path: DUO_ANN/instances_test.json
  dataset_dir: dataset/DUO
  allow_empty: true

···

```
#### Train PP-YOLOE

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus "0,1,2,3"
  tools/train.py -c configs/ppyoloe/ppyoloe_crn_l_36e_coco.yml --eval
```

#### Train YOLOv10

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus "0,1,2,3"
  tools/train.py -c configs/yolov10/yolov10_l_36e_coco.yml --eval
```

### How to train PP-YOLOE-E and YOLOv10-E with E-ConvNeXt backbone

Configure the data as in the previous section.

#### Train PP-YOLOE with E-ConvNeXt backbone

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus "0,1,2,3"
  tools/train.py -c configs/ppyoloe_E_ConvNeXt/ppyoloe_l_cspconvnext_tiny_36e_coco.yml --eval
```

#### Train YOLOv10 with E-ConvNeXt backbone

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus "0,1,2,3"
  tools/train.py -c configs/yolov10_E_ConvNeXt/yolov10_l_cspconvnext_tiny.yml --eval
```


