import tensorflow as tf
import numpy as np

def Param():
    flags = tf.app.flags
    
    ## Param
    flags.DEFINE_integer("epoch",         40,           "Epoch to train [25]")
    flags.DEFINE_integer("c_epoch",       0,            "current Epoch")
    flags.DEFINE_integer("enhance",       5,            "Enhancement for different matrix")
    flags.DEFINE_float("lr",              0.0002,       "Learning rate of for adam [0.0002]")
    flags.DEFINE_float("beta1",           0.5,          "Momentum term of adam [0.5]")
    flags.DEFINE_float("side_D",          1.0,          "side discriminator for cycle updating")
    flags.DEFINE_float("cycle",           0.5,          "threshold for cycle updating")
        
    ## Data
    flags.DEFINE_string("dataset",        "loam",       "The name of dataset [celebA, mnist, loam, lsun]")
    flags.DEFINE_integer("sample_size",   64,           "The number of sample images [64]")
    flags.DEFINE_integer("img_dim",       3,            "Dimension of image color. [3]")
    flags.DEFINE_integer("code_dim",      512,          "code dimension")
    
    ## Dir
    flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
    flags.DEFINE_string("sample_dir",     "samples",    "Directory name to save the image samples [samples]")
    flags.DEFINE_string("result_dir",     "results",    "Directory name to save SeqSLAM results [results]")
    flags.DEFINE_string("log_dir",        "logs",       "Directory name to save logs [logs]")
    flags.DEFINE_string("model_dir",      "ALI_CYC",    "Model selected for both model saving and ")
    flags.DEFINE_string("data_dir",       "data",       "Directory name to extract image datas")
    
    ## Training
    flags.DEFINE_string("method",         "ALI_CYC",    "ALI or ALI_CYC")
    flags.DEFINE_integer("sample_step",   2,            "The interval of generating sample. [500]")
    flags.DEFINE_integer("save_step",     100,          "The interval of saveing checkpoints. [500]")
    flags.DEFINE_integer("img_filter",    64,           "The number of image filters")
    flags.DEFINE_integer("dX_dim",        1024,         "The number of discriminator for image")
    flags.DEFINE_integer("dZ_dim",        1024,         "The number of discriminator for code")
    flags.DEFINE_integer("dJ_dim",        2048,         "The number of discriminator for Joint")
    flags.DEFINE_integer("image_size",    500,          "The size of image to use (will be center cropped) [108]")
    flags.DEFINE_integer("output_size",   64,           "The size of the output images to produce [64]")
    flags.DEFINE_integer("train_size",    np.inf,       "The size of train images [np.inf]")
    flags.DEFINE_integer("batch_size",    64,           "The number of batch images [64]")

    ## SeqSLAM
    flags.DEFINE_float("v_ds",            10,           "seqslam distance")
    flags.DEFINE_float("vmin",            0.8,          "min velocity of seqslam")
    flags.DEFINE_float("vskip",           0.1,          "velocity gap")
    flags.DEFINE_float("vmax",            1.2,          "max velocity of seqslam")
    flags.DEFINE_integer("Rwindow",       10,           "rainbow")
    flags.DEFINE_integer("test_len",      1000,          "test data length")
    flags.DEFINE_string("test_dir",    "test_T10_R2.5",   "Directory name to extract image datas")

    ## Flag
    flags.DEFINE_boolean("is_train",      False,        "True for training, False for testing [False]")
    flags.DEFINE_boolean("is_crop",       True,         "True for training, False for testing [False]")
    flags.DEFINE_boolean("restore",       False,        "restore from pre trained")
    flags.DEFINE_boolean("visualize",     False,        "True for visualizing, False for nothing [False]")

    return flags.FLAGS
