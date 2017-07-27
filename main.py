import os
import sys
import tensorflow as tf
from parameters import *
from src.model.model import Net
from src.model.model3D import Net3D
from src.model.model_feature import Net_Feature
from src.model.model_simpleCYC import Net_simpleCYC

# Obtain parameters
args = Param()

def main(_):
    
    # check the existence of directories
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.tb_dir):
        os.makedirs(args.tb_dir)


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

    if args.is_train == False:
        args.batch_size = 1

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth=True

    #GPUID = 0
    #gpuNow = '/gpu:'+str(GPUID)
    #with tf.device(gpuNow):

    with tf.Session(config=config) as sess:
        model = Net_model(sess, args)
        model.train(args) if args.is_train == True else model.test(args)

if __name__ == '__main__':
    tf.app.run()

