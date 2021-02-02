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

I have tested this code on ubuntu16.04+cuda9.0+pytorch1.1+python3.6+spconv1.0

The pretrained weights are in the output/kitti_models/3DSSD_openPCDet/3DSSD/ckpt/.
The eval performance on the Car class is as follows:
```
Car AP@0.70, 0.70, 0.70:
bbox AP:96.5468, 90.0235, 89.4066
bev  AP:90.3444, 88.0784, 86.0698
3d   AP:89.2219, 78.8593, 77.5890
aos  AP:96.52, 89.95, 89.25
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.2011, 95.0305, 92.6650
bev  AP:93.2919, 89.1952, 88.1910
3d   AP:91.4331, 82.2283, 77.8059
aos  AP:98.18, 94.93, 92.49
Car AP@0.70, 0.50, 0.50:
bbox AP:96.5468, 90.0235, 89.4066
bev  AP:96.6237, 90.1257, 89.6772
3d   AP:96.5594, 90.0998, 89.6259
aos  AP:96.52, 89.95, 89.25
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.2011, 95.0305, 92.6650
bev  AP:98.3041, 95.4983, 95.0182
3d   AP:98.2703, 95.3970, 94.8667
aos  AP:98.18, 94.93, 92.49
```

