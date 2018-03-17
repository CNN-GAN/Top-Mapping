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
tensorboard --logdir=./logs --port=8080
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

## 3D CNN framework
* Add PCD data loader **(done)**
* Add 3D module **(done)**

## Loss Update
* use Wessentian GAN to update the Loss **(done)**

# Update Notes:
## 2017/07/10 10:53:42 CST
Update model.py, to save difference matrix for latter use.

## 2017/07/12 19:02:12 CST
Add gtav_cls for gtav scene classification

## 2017/07/17 9:07:12 CST
Add simpleCYC model, this work is based on the alpha-GAN and conditional GAN.
We extract the weather condition invariant code from different style images, and use the latent code 
to do the LCD job.

# Install FFmpeg
sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update
sudo apt-get install ffmpeg
sudo apt-get install frei0r-plugins

# Use FFmpeg to generate video
ffmpeg -framerate 25 -i %05d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
