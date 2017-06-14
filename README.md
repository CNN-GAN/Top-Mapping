# Top Mapping

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

# TODO
## Basic
* generate the test lidar-map under gaussian noise in both translation and rotation and test the result. **(done)**
* test the matching result on the the GT maps and noise-Maps. **(done)**
* Caculate the Precious-Recall Curve **(done)**
* update current framework **(done)**

## Check other method
* check ALI-GAN **(done)**
* check on different iteration **(done)**
* check on different translation and rotation error **(done)**
* use Wessentian GAN to update the Loss **(doing)**

## Enhance to the SeqSLAM approach
* add SeqSLAM framework into the current **(done)**
* check SeqSLAM result **(done)**
* check dataset on originl SeqSLAM method **(done)**
* Add KNN search to accelerate the matching speed **(done)**

## Dynamic Mapping
* Update the dynamic mapping framework **(doing)**
* Add 3D module **(doing)**
