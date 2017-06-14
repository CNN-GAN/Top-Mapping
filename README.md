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

## Check the translation and rotation invarit ability
* generate the test lidar-map under gaussian noise in both translation and rotation and test the result. **(done)**
* test the matching result on the the GT maps and noise-Maps. **(done)**
* Caculate the Precious-Recall Curve

## Check other method
* check ALI-GAN **(done)**
* check on different iteration
* check on different translation and rotation error.

## Enhance to the SeqSLAM approach
* add SeqSLAM framework into the current **(done)**
* check SeqSLAM result **(done)**
* check dataset on originl SeqSLAM method **(done)**
* Add KNN search to accelerate the matching speed

## Basic
* update current framework