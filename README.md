# Top Mapping

# Required
## Tensorflow 1.2
```bash
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0-cp27-none-linux_x86_64.whl
```

# Running
## ALI features
```bash
python main.py
```

## Tensorboard
```bash
tensorboard --logdir=./logs --port=8008
```

## Param
In the file parameters.py

# DONE
## Basic
* generate the test lidar-map under gaussian noise in both translation and rotation and test the result. **(done)**
* test the matching result on the the GT maps and noise-Maps. **(done)**
* Caculate the Precious-Recall Curve **(done)**
* update current framework **(done)**

## Check other method
* check ALI-GAN **(done)**
* check on different iteration **(done)**
* check on different translation and rotation error **(done)**


## Enhance to the SeqSLAM approach
* add SeqSLAM framework into the current **(done)**
* check SeqSLAM result **(done)**
* check dataset on originl SeqSLAM method **(done)**
* Add KNN search to accelerate the matching speed **(done)**

## Dynamic Mapping
* Update the dynamic mapping framework **(done)**
* Update save module, able to save pointcloud, image and pose **(done)**

# TODO
## 3D CNN framework
* Add 3D module **(doing)**


## Loss Update
* use Wessentian GAN to update the Loss **(doing)**
