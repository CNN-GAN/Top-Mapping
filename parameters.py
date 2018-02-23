import tensorflow as tf
import numpy as np

def Param():
    flags = tf.app.flags
    
    ## Param
    flags.DEFINE_integer("epoch",              40,           "Epoch to train [40]")
    flags.DEFINE_integer("iter_num",           1000000,      "iteration to train [40]")
    flags.DEFINE_integer("c_epoch",            0,            "current Epoch")
    flags.DEFINE_integer("get_epoch",          15,           "get features from Epoch")

    flags.DEFINE_integer("enhance",            20,           "Enhancement for different matrix")
    flags.DEFINE_float("lr",                   0.0002,       "Learning rate of for adam [0.0002]")
    flags.DEFINE_float("beta1",                0.5,          "Momentum term of adam [0.5]")
    flags.DEFINE_float("side_D",               0.1,          "side discriminator for cycle updating")
    flags.DEFINE_float("cycle",                0.1,          "threshold for cycle updating")
    flags.DEFINE_float("head_diff",            0.1,          "threshold for cycle updating")
    flags.DEFINE_float("maha",                 0.5,          "threshold for cycle updating")
    flags.DEFINE_float("trans",                0.2,          "threshold for cycle updating")
    flags.DEFINE_float("in_cycle",             1.0,          "threshold for inner cycle updating")
    flags.DEFINE_float("distance_weighting",   1.0,     "threshold for far/near frames")
        
    ## Data
    flags.DEFINE_string("dataset",        "NCTL",   "The name of dataset [new_loam, NCTL, GTAV, nordland]")
    flags.DEFINE_string("date_format",    "%m.%d_%H-%M",   "Date format")
    flags.DEFINE_integer("sample_size",   64,           "The number of sample images [64]")
    flags.DEFINE_integer("img_dim",       3,            "Dimension of image color. [3]")
    flags.DEFINE_integer("output_size",   64,           "The size of the output images to produce [64]")
    flags.DEFINE_integer("code_dim",      512,          "code dimension")
    flags.DEFINE_integer("condition_dim", 64,           "condition code dimension")
    
    ## Dir
    flags.DEFINE_string("run_id_string",  "",           "run_id_string")
    flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
    flags.DEFINE_string("result_dir",     "result",    "Directory name to save SeqSLAM results [results]")
    flags.DEFINE_string("data_dir",       "data",       "Directory name to extract image datas")
    flags.DEFINE_string("log_dir",        "logs",       "Directory name to save tensorboard [tb_logs]")
    flags.DEFINE_string("sample_dir",     "logs/samples",    "Directory name to save the image samples [samples]")
    flags.DEFINE_string("model_date",     "01.03_20-20",      "Directory name to save tensorboard [tb_logs]")

    flags.DEFINE_string("new_loam_img",   "01.03_20-20",      "Directory name to save tensorboard [tb_logs]")
    flags.DEFINE_string("new_loam_pcd",   "12.03_09-03",      "Directory name to save tensorboard [tb_logs]")
    flags.DEFINE_string("nctl_img",       "12.08_23-19",      "Directory name to save tensorboard [tb_logs]")
    flags.DEFINE_string("nctl_pcd",       "12.08_23-18",      "Directory name to save tensorboard [tb_logs]")

    flags.DEFINE_integer("img_epoch",     15,           "The size of the output images to produce [64]")
    flags.DEFINE_integer("pcd_epoch",     15,           "The size of the output images to produce [64]")
    
    flags.DEFINE_string("log_notes",      "headingInv model, with frame_skip 10",      "logs")

    ## Training
    flags.DEFINE_string("method",         "ALI_CLC",    "BiGAN_GTAV, conditionCYC, simpleCYC, ALI_CLC, ALI or ALI_IV, Reweight, headingInv")
    flags.DEFINE_string("Search",         "N",          "N normal, A ann")
    flags.DEFINE_string("Loss",           "LSGAN",      "WGAN, LSGAN")
    flags.DEFINE_float("scale",           0.1,          "Scale for WGAN")
    flags.DEFINE_integer("sample_step",   2,            "The interval of generating sample. [500]")
    flags.DEFINE_integer("save_step",     100,         "The interval of saveing checkpoints. [500]")
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

    ## Mahanobis 
    flags.DEFINE_integer("frame_near",    2,           "Frames to skip for mahanobis")    
    flags.DEFINE_integer("frame_far",     8,           "Frames to skip for mahanobis")    

    ## 3D conv
    flags.DEFINE_integer("voxel_filter",  64,           "The number of image filters")
    flags.DEFINE_integer("voxel_size",    64,           "Set map scale [128, 128, 16]")
    flags.DEFINE_integer("voxel_dim",     1,            "voxel map dim")
    flags.DEFINE_integer("voxel_code",    128,          "voxel code dimension")

    ## SeqSLAM
    flags.DEFINE_float("v_ds",            10,            "seqslam distance")
    #flags.DEFINE_float("enhance",        20,            "enhance distance")
    flags.DEFINE_float("vmin",            0.8,           "min velocity of seqslam")
    flags.DEFINE_float("vskip",           0.1,           "velocity gap")
    flags.DEFINE_float("vmax",            1.2,           "max velocity of seqslam")
    flags.DEFINE_integer("Rwindow",       10,            "rainbow")
    flags.DEFINE_integer("frame_skip",    4,             "frame skip")    
    flags.DEFINE_integer("Knn",           5,             "K nearest point")
    flags.DEFINE_integer("test_len",      200,           "test data length")
    flags.DEFINE_integer("test_base",     0,             "test data base")
    flags.DEFINE_string("test_dir",    "test_T10_R2",    "Directory name to extract image datas")
    flags.DEFINE_string("match_method",   "ANN",         "ANN or Force")
    flags.DEFINE_float("match_distance",   10,           "match threshold for distance")
    flags.DEFINE_float("match_thres",      80,           "match threshold for GTAV")

    ## Flag
    flags.DEFINE_boolean("is_3D",             False,        "True for train the 3D module")
    flags.DEFINE_boolean("is_train",          False,        "True for training, False for testing [False]")
    flags.DEFINE_boolean("is_reconstruct",    False,        "True for reconstruct")
    flags.DEFINE_boolean("is_obtain_feature", False,        "True for obtain features from sequence")
    flags.DEFINE_boolean("is_crop",           True,         "True for crop image")
    flags.DEFINE_boolean("restore",           False,        "restore from pre trained")
    flags.DEFINE_boolean("visualize",         False,        "True for visualizing, False for nothing [False]")
    
    ## Origional SeqSLAM
    flags.DEFINE_boolean("SeqSLAM",       False,        "SeqSLAM")
    flags.DEFINE_boolean("SeqGTAV",       False,        "SeqGTAV")
    flags.DEFINE_boolean("SeqVGG",        False,        "SeqVGG")

    ## Plotting
    flags.DEFINE_boolean("plot",           True,         "True for ploting figures")

    ## Plot for paper 1
    flags.DEFINE_boolean("plot_3D",        False,        "True for ploting 3D")
    flags.DEFINE_boolean("plot_2D",        True,         "True for ploting 2D")
    flags.DEFINE_boolean("plot_joint",     False,        "True for ploting Joint")
    flags.DEFINE_boolean("plot_paper1",    False,        "True for ploting paper1")

    ## Plot for paper 2
    flags.DEFINE_boolean("plot_slfl",      False,        "True for ploting Joint")
    flags.DEFINE_boolean("plot_paper2",    False,        "True for ploting paper2")

    ## Plot for paper 3
    flags.DEFINE_boolean("plot_simplecyc", False,         "True for ploting simplecyc")
    flags.DEFINE_boolean("plot_VGG",       False,         "True for VGG")
    flags.DEFINE_boolean("plot_biganGTAV", False,         "True for biganGTAV")
    flags.DEFINE_boolean("plot_paper3",    False,         "True for ploting paper3")

    return flags.FLAGS
