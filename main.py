import os
import sys
import tensorflow as tf

from time import strftime
from parameters import *
from src.model.model import Net
from src.model.model3D import Net3D
from src.model.model_feature import Net_Feature
from src.model.model_simpleCYC import Net_simpleCYC
from src.model.model_BIGAN_GTAV import Net_BIGAN_GTAV
from src.model.model_reweight import Net_REWEIGHT
from src.model.model_headingInv import Net_HEADINGINV

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
from src.plot.CFL.plotDraw_biganGTAV import Plot_biganGTAV

# SeqSLAM for LiDAR inputs
from src.third_function.pyseqslam.seq_lidar import Seq_LiDAR
# SeqSLAM for GTAV image
from src.third_function.pyseqslam.seq_gtav import Seq_GTAV
# SeqSLAM with VGG cnn features
from src.third_function.vgg16.vgg16 import Seq_VGG

os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Obtain parameters
args = Param()

def main(_):
    
    # Current model string
    if args.is_3D == True:
        args.method_path = 'ALI_3D'
    else:
        args.method_path = args.method

    if args.is_train == True:
        args.run_id_string = "{}/{}".format(args.method_path, strftime(args.date_format))
        with open('log.txt', 'a') as the_file:
            the_file.write(args.run_id_string)
            the_file.write(args.log_notes) 
    else:
        if args.dataset == "new_loam":
            if args.method == "ALI":
                args.run_id_string = "{}/{}".format(args.method_path, strftime(args.ali_kitti_img))
            else:
                args.run_id_string = "{}/{}".format(args.method_path, strftime(args.aliclc_kitti_img))
        else:
            if args.method == "ALI":
                args.run_id_string = "{}/{}".format(args.method_path, strftime(args.ali_nctl_img))
            else:
                args.run_id_string = "{}/{}".format(args.method_path, strftime(args.aliclc_nctl_img))


    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.run_id_string)
    args.sample_dir = os.path.join(args.sample_dir, args.dataset, args.run_id_string)
    args.result_dir = os.path.join(args.result_dir, args.dataset, args.run_id_string)
    args.log_dir = os.path.join(args.log_dir, args.dataset, args.run_id_string)

    # check the existence of directories
    if args.is_train == True:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        if not os.path.exists(args.sample_dir):
            os.makedirs(args.sample_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
    else:
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

    # set gpu usage
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth=True

    # Reweighting module
    if args.method == 'Reweight':

        args.v_img_path = os.path.join('result', args.dataset, 'ALI_CLC', args.img_date)
        args.v_pcd_path = os.path.join('result', args.dataset, 'ALI_3D',  args.pcd_date)

        if not os.path.exists(args.v_img_path):
            print ('no such img code {}'.format(args.v_img_path))
            return

        if not os.path.exists(args.v_pcd_path):
            print ('no such pcd code {}'.format(args.v_pcd_path))
            return

        with tf.Session(config=config) as sess:
            model = Net_REWEIGHT(sess, args)
            if args.is_train == True:
                model.train(args) 
            else:
                model.test(args)                 
        return

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
        if args.plot_biganGTAV == True:
            Plot_biganGTAV(args)
        if args.plot_slfl == True:
            Plot_SLFL(args)

        if args.plot_paper1 == True:
            Plot_Paper1(args)
        if args.plot_paper2 == True:
            Plot_Paper2(args)
        if args.plot_paper3 == True:
            Plot_Paper3(args)

        return

    if args.is_3D == True:
        Net_model = Net3D
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

    if args.method == 'headingInv':
        Net_model = Net_HEADINGINV

    if args.is_train == False:
        args.batch_size = 1

    with tf.Session(config=config) as sess:

        # Use VGG CNN features for SeqSLAM
        if args.SeqVGG == True:
            Seq_VGG(sess, args)
            return

        model = Net_model(sess, args)
        if args.is_train == True:
            model.train(args) 
            return
            
        if args.is_reconstruct == True:
            model.reconstruct(args)
            return

        if args.is_obtain_feature == True:
            model.generate_codes(args)
            return

        model.test(args)
        return

if __name__ == '__main__':
    tf.app.run()
