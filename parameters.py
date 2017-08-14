import tensorflow as tf
import numpy as np

def Param():
    flags = tf.app.flags
    
    ## Param
    flags.DEFINE_integer("epoch",         40,           "Epoch to train [40]")
    flags.DEFINE_integer("c_epoch",       0,            "current Epoch")
    flags.DEFINE_integer("enhance",       20,           "Enhancement for different matrix")
    flags.DEFINE_float("lr",              0.0002,       "Learning rate of for adam [0.0002]")
    flags.DEFINE_float("beta1",           0.5,          "Momentum term of adam [0.5]")
    flags.DEFINE_float("side_D",          0.1,          "side discriminator for cycle updating")
    flags.DEFINE_float("cycle",           0.5,          "threshold for cycle updating")
    flags.DEFINE_float("in_cycle",        1.0,          "threshold for inner cycle updating")
        
    ## Data
    flags.DEFINE_string("dataset",        "new_loam",   "The name of dataset [new_loam, GTAV, loam]")
    flags.DEFINE_integer("sample_size",   64,           "The number of sample images [64]")
    flags.DEFINE_integer("img_dim",       3,            "Dimension of image color. [3]")
    flags.DEFINE_integer("output_size",   64,           "The size of the output images to produce [64]")
    flags.DEFINE_integer("code_dim",      512,          "code dimension")
    flags.DEFINE_integer("condition_dim", 64,           "condition code dimension")
    
    ## Dir
    flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
    flags.DEFINE_string("sample_dir",     "logs/samples",    "Directory name to save the image samples [samples]")
    flags.DEFINE_string("result_dir",     "logs/results",    "Directory name to save SeqSLAM results [results]")
    flags.DEFINE_string("data_dir",       "data",       "Directory name to extract image datas")
    flags.DEFINE_string("log_dir",        "logs",       "Directory name to save tensorboard [tb_logs]")
    flags.DEFINE_string("log_name",       "ALI",        "Directory name to save tensorboard [tb_logs]")
    
    ## Training
    flags.DEFINE_string("method",         "simpleCYC",        "conditionCYC, simpleCYC, ALI_CLC, ALI or ALI_IV")
    flags.DEFINE_string("Search",         "N",          "N normal, A ann")
    flags.DEFINE_string("Loss",           "LSGAN",      "WGAN, LSGAN")
    flags.DEFINE_float("scale",           0.1,          "Scale for WGAN")
    flags.DEFINE_integer("sample_step",   1,            "The interval of generating sample. [500]")
    flags.DEFINE_integer("save_step",     50,           "The interval of saveing checkpoints. [500]")
    flags.DEFINE_integer("img_filter",    64,           "The number of image filters")
    flags.DEFINE_integer("dX_dim",        1024,         "The number of discriminator for image")
    flags.DEFINE_integer("dZ_dim",        1024,         "The number of discriminator for code")
    flags.DEFINE_integer("dJ_dim",        2048,         "The number of discriminator for Joint")
    flags.DEFINE_integer("image_size",    500,          "The size of image to use (will be center cropped) [108]")
    flags.DEFINE_integer("train_size",    np.inf,       "The size of train images [np.inf]")
    flags.DEFINE_integer("batch_size",    64,           "The number of batch images [64]")
    flags.DEFINE_integer("d_iter",        4,            "The number of iteration for discriminator")
    flags.DEFINE_integer("g_iter",        8,            "The number of iteration for generator")
    flags.DEFINE_integer("iteration",     10000000,     "Training iteration")

    ## 3D conv
    flags.DEFINE_integer("voxel_filter",  64,           "The number of image filters")
    flags.DEFINE_integer("voxel_size",    64,           "Set map scale [128, 128, 16]")
    flags.DEFINE_integer("voxel_dim",     1,            "voxel map dim")
    flags.DEFINE_integer("voxel_code",    512,          "voxel code dimension")

    ## SeqSLAM
    flags.DEFINE_float("v_ds",            10,            "seqslam distance")
    #flags.DEFINE_float("enhance",        20,            "enhance distance")
    flags.DEFINE_float("vmin",            0.8,           "min velocity of seqslam")
    flags.DEFINE_float("vskip",           0.1,           "velocity gap")
    flags.DEFINE_float("vmax",            1.2,           "max velocity of seqslam")
    flags.DEFINE_integer("Rwindow",       10,            "rainbow")
    flags.DEFINE_integer("frame_skip",    3,             "frame skip")    
    flags.DEFINE_integer("Knn",           5,             "K nearest point")
    flags.DEFINE_integer("test_len",      300,           "test data length")
    flags.DEFINE_string("test_dir",    "test_T10_R2",    "Directory name to extract image datas")
    flags.DEFINE_string("match_method",   "ANN",         "ANN or Force")
    flags.DEFINE_float("match_distance",   10,           "match threshold for distance")
    flags.DEFINE_float("match_thres",      40,           "match threshold for GTAV")

    ## Flag
    flags.DEFINE_boolean("is_3D",         False,        "True for train the 3D module")
    flags.DEFINE_boolean("is_train",      False,        "True for training, False for testing [False]")
    flags.DEFINE_boolean("is_crop",       True,         "True for crop image")
    flags.DEFINE_boolean("restore",       False,        "restore from pre trained")
    flags.DEFINE_boolean("visualize",     False,        "True for visualizing, False for nothing [False]")
    
    ## Origional SeqSLAM
    flags.DEFINE_boolean("SeqSLAM",       False,        "SeqSLAM")
    flags.DEFINE_boolean("SeqGTAV",       False,        "SeqGTAV")

    ## Plotting
    flags.DEFINE_boolean("plot",           True,         "True for ploting figures")
    ## Plot for paper 1
    flags.DEFINE_boolean("plot_3D",        False,        "True for ploting 3D")
    flags.DEFINE_boolean("plot_2D",        False,         "True for ploting 2D")
    flags.DEFINE_boolean("plot_joint",     False,        "True for ploting Joint")
    flags.DEFINE_boolean("plot_paper1",    False,        "True for ploting paper1")
    ## Plot for paper 2
    flags.DEFINE_boolean("plot_slfl",      False,        "True for ploting Joint")
    flags.DEFINE_boolean("plot_paper2",    False,         "True for ploting paper2")
    ## Plot for paper 3
    flags.DEFINE_boolean("plot_simplecyc", True,         "True for ploting simplecyc")

    return flags.FLAGS
