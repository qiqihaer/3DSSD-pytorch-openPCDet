# 3DSSD-pytorch-openPCDet
The successful implementation of 3DSSD in Pytorch

Thanks for the [`[OpenPCDet]`](https://github.com/open-mmlab/OpenPCDet)!!!
This implementation of the 3DSSD is mainly based on the pcdet v0.3.

## Preparation
1. Clone this repository

2. Install the Python dependencies.
```
pip install -r requirements.txt
```

3. Option: Install the spconv. Please follow the instructions in the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md).  This 3DSSD does not use spconv. However, for better use the OpenPCDet, I recommend to install it.

4. Install the pcdet library.
```
python setup.py develop
```

5. Install the pointnet2_3DSSD libarary.
```
cd pcdet/ops/pointnet2/pointnet2_3DSSD/
python setup.py develop
```

## Train a Model

I have set the default config file to the 3DSSD model. So, you can just run:
```
cd tools
python train.py
```

The trainning log and tensorboard log are saved into output dir

## Eval the model
You can follow the instructions in the [openPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md). Please set the --cfg_file to 'cfgs/kitti_models/3DSSD_openPCDet.yaml'

## Pretrained Weights
The logs in the output dir.

