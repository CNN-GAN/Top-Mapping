import os
import sys
import tensorflow as tf
from parameters import *
from model import Net
from model3D import Net3D

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
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.is_train == False:
        args.batch_size = 1

    if args.is_3D == True:
        Net_model = Net
    else:
        Net_model = Net3D

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        model = Net_model(sess, args)
        model.train(args) if args.is_train == True else model.test(args)

if __name__ == '__main__':
    tf.app.run()
