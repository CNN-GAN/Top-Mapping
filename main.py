import os
import sys
import tensorflow as tf
from parameters import *
from src.model.model import Net
from src.model.model3D import Net3D
from src.model.model_feature import Net_Feature
from src.model.model_simpleCYC import Net_simpleCYC
from src.model.model_BIGAN_GTAV import Net_BIGAN_GTAV

# For the first paper, Unsupervised LiDAR Feature Learning
from src.plot.ULFL.plotDraw_joint import Plot_Joint
from src.plot.ULFL.plotDraw_3d import Plot_3D
from src.plot.ULFL.plotDraw_2d import Plot_2D
from src.plot.plotPaper1  import Plot_Paper1
from src.plot.plotPaper2  import Plot_Paper2
from src.plot.plotPaper3  import Plot_Paper3

# For the second paper, Stable LiDAR Feature Learning
from src.plot.SLFL.plotDraw_slfl import Plot_SLFL

# For the third paper, Common Feature Learning
from src.plot.CFL.plotDraw_simpleCYC import Plot_simpleCYC
from src.plot.CFL.plotDraw_VGG import Plot_VGG

# SeqSLAM for LiDAR inputs
from src.third_function.pyseqslam.seq_lidar import Seq_LiDAR
# SeqSLAM for GTAV image
from src.third_function.pyseqslam.seq_gtav import Seq_GTAV
# SeqSLAM with VGG cnn features
from src.third_function.vgg16.vgg16 import Seq_VGG

# Obtain parameters
args = Param()

os.environ["CUDA_VISIBLE_DEVICES"]="2"

def main(_):

    # Original SeqSLAM method
    if args.SeqSLAM == True:
        Seq_LiDAR(args)
        return

    if args.SeqGTAV == True:
        Seq_GTAV(args)
        return

    if args.plot == True:
        print ("ploting the figures...")

        if args.plot_3D == True:
            Plot_3D(args)
        if args.plot_2D == True:
            Plot_2D(args)
        if args.plot_joint == True:
            Plot_Joint(args)
        if args.plot_simplecyc == True:
            Plot_simpleCYC(args)
        if args.plot_VGG == True:
            Plot_VGG(args)
        if args.plot_slfl == True:
            Plot_SLFL(args)

        if args.plot_paper1 == True:
            Plot_Paper1(args)
        if args.plot_paper2 == True:
            Plot_Paper2(args)
        if args.plot_paper3 == True:
            Plot_Paper3(args)

        return

    # check the existence of directories
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    log_path = os.path.join(args.log_dir, args.log_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)


    if args.is_3D == True:
        Net_model = Net3D
        args.dataset = 'new_loam'
        args.batch_size = 64
    else:
        Net_model = Net
    
    if args.method == 'conditionCYC':
        Net_model = Net_Feature
        args.dataset = 'GTAV'

    if args.method == 'simpleCYC':
        Net_model = Net_simpleCYC
        args.dataset = 'GTAV'

    if args.method == 'BiGAN_GTAV':
        Net_model = Net_BIGAN_GTAV
        args.dataset = 'GTAV'

    if args.is_train == False:
        args.batch_size = 1

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:

        # Use VGG CNN features for SeqSLAM
        if args.SeqVGG == True:
            Seq_VGG(sess, args)
            return

        model = Net_model(sess, args)
        if args.is_train == True:
            model.train(args) 
        else:
            if args.is_reconstruct == True:
                model.reconstruct(args)
            else:
                model.test(args)

if __name__ == '__main__':
    tf.app.run()

