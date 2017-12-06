import os
import numpy as np
from src.util.utils import *

class DataSampler(object):
    def __init__(self, args, data_files):
        self.shape = [args.output_size, args.output_size, args.img_dim]
        shuffle(data_files)
        self.data_files = data_files
        self.batch_len = min(len(data_files), args.train_size) // args.batch_size
        self.current_batch = 0
        self.batch_size = args.batch_size
        self.is_crop = args.is_crop
        self.output_size = args.output_size
        self.image_size = args.image_size

    def load_new_data(self):
        ### Get datas ###
        batch_files  = self.data_files[self.current_batch * self.batch_size : (self.current_batch+1)*self.batch_size]
        ## get real images
        batch        = [get_image(batch_file, self.image_size, is_crop=self.is_crop, \
                                  resize_w=self.output_size, is_grayscale = 0) \
                        for batch_file in batch_files]
        batch_images = np.array(batch).astype(np.float32)
        return batch_images

    def __call__(self):
        if self.current_batch >= self.batch_len:
            shuffle(self.data_files)
            self.current_batch = 0

        batch_images = self.load_new_data()
        self.current_batch += 1            
        return batch_images


class TriDataSampler(object):
    def __init__(self, args, v_img, v_pcd, v_pose):
        self.shape = [args.output_size, args.output_size, args.img_dim]
        self.v_img  = v_img
        self.v_pcd  = v_pcd        
        self.v_pose = v_pose        
        self.batch_len = len(v_img)
        self.fn = args.frame_near
        self.ff = args.frame_far
        self.current_id = self.ff

        self.batch_size = 1
        self.is_crop = args.is_crop
        self.output_size = args.output_size
        self.image_size = args.image_size

    def load_new_data(self):
        ### Get datas ###
        self.cc_img = self.v_img[current_id]
        self.cc_pcd = self.v_pcd[current_id]
        
        self.cn_img = self.v_img[current_id-self.fn]
        self.cn_pcd = self.v_pcd[current_id-self.fn]

        self.cf_img = self.v_img[current_id-self.ff]
        self.cf_pcd = self.v_pcd[current_id-self.ff]


    def __call__(self):
        if self.current_id >= self.batch_len:
            self.current_id = self.ff

        self.load_new_data()
        self.current_id += 1            
        return self.cc_img, self.cc_pcd, self.cn_img, self.cn_pcd, self.cf_img, self.cf_pcd

class NoiseSampler(object):
    def __init__(self, args):
        self.sample_size = args.sample_size
        self.code_dim = args.code_dim

    def __call__(self):
        return np.random.normal(loc=0.0, scale=1.0, \
                                        size=(self.sample_size, self.code_dim)).astype(np.float32)
