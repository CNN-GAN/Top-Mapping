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


class HeadINVdataSampler(object):
    def __init__(self, args, rn2_files, rn1_files, r0_files, rp1_files, rp2_files):
        self.rn2  = rn2_files
        self.rn1  = rn1_files
        self.r0   = r0_files
        self.rp1  = rp1_files
        self.rp2  = rp2_files

        self.batch_len = len(r0_files) // (args.batch_size * 2)
        self.current_batch = 0

        self.bag_ids = np.arange(len(r0_files))
        np.random.shuffle(self.bag_ids)
        print("[*] Dataset shuffled!")

        self.batch_size = args.batch_size
        self.is_crop = args.is_crop
        self.output_size = args.output_size
        self.image_size = args.image_size

    def load_new_data(self):
        ### Get datas ###
        data_rn2 = []
        data_rn1 = []
        data_r0 = []
        data_rp1 = []
        data_rp2 = []

        start_id = self.current_batch * 2 * self.batch_size
        for idx in range(self.batch_size):
            data_idx =  self.bag_ids[start_id+idx]
            data_rn2 += [self.rn2[data_idx]]
            data_rn1 += [self.rn1[data_idx]]
            data_r0  += [self.r0[data_idx]]
            data_rp1 += [self.rp1[data_idx]]
            data_rp2 += [self.rp2[data_idx]]

        data_orn2 = []
        data_orn1 = []
        data_or0 = []
        data_orp1 = []
        data_orp2 = []

        start_id = (self.current_batch * 2 + 1) * self.batch_size
        for idx in range(self.batch_size):
            data_idx =  self.bag_ids[start_id+idx]
            data_orn2 += [self.rn2[data_idx]]
            data_orn1 += [self.rn1[data_idx]]
            data_or0  += [self.r0[data_idx]]
            data_orp1 += [self.rp1[data_idx]]
            data_orp2 += [self.rp2[data_idx]]
                    
        ## get real images
        batch_rn2        = [get_image(batch_file, self.image_size, is_crop=self.is_crop, \
                                      resize_w=self.output_size, is_grayscale = 0) \
                            for batch_file in data_rn2]
        self.batch_rn2_images = np.array(batch_rn2).astype(np.float32)
        
        batch_rn1        = [get_image(batch_file, self.image_size, is_crop=self.is_crop, \
                                      resize_w=self.output_size, is_grayscale = 0) \
                            for batch_file in data_rn1]
        self.batch_rn1_images = np.array(batch_rn1).astype(np.float32)
        
        batch_r0         = [get_image(batch_file, self.image_size, is_crop=self.is_crop, \
                                      resize_w=self.output_size, is_grayscale = 0) \
                            for batch_file in data_r0]
        self.batch_r0_images  = np.array(batch_r0).astype(np.float32)
        
        batch_rp1        = [get_image(batch_file, self.image_size, is_crop=self.is_crop, \
                                      resize_w=self.output_size, is_grayscale = 0) \
                            for batch_file in data_rp1]
        self.batch_rp1_images = np.array(batch_rp1).astype(np.float32)
        
        batch_rp2        = [get_image(batch_file, self.image_size, is_crop=self.is_crop, \
                                      resize_w=self.output_size, is_grayscale = 0) \
                            for batch_file in data_rp2]
        self.batch_rp2_images = np.array(batch_rp2).astype(np.float32)
        
        ## get other images
        batch_orn2        = [get_image(batch_file, self.image_size, is_crop=self.is_crop, \
                                       resize_w=self.output_size, is_grayscale = 0) \
                             for batch_file in data_orn2]
        self.batch_orn2_images = np.array(batch_orn2).astype(np.float32)
        
        batch_orn1        = [get_image(batch_file, self.image_size, is_crop=self.is_crop, \
                                       resize_w=self.output_size, is_grayscale = 0) \
                             for batch_file in data_orn1]
        self.batch_orn1_images = np.array(batch_orn1).astype(np.float32)
        
        batch_or0         = [get_image(batch_file, self.image_size, is_crop=self.is_crop, \
                                       resize_w=self.output_size, is_grayscale = 0) \
                             for batch_file in data_or0]
        self.batch_or0_images  = np.array(batch_or0).astype(np.float32)
        
        batch_orp1        = [get_image(batch_file, self.image_size, is_crop=self.is_crop, \
                                       resize_w=self.output_size, is_grayscale = 0) \
                             for batch_file in data_orp1]
        self.batch_orp1_images = np.array(batch_orp1).astype(np.float32)
        
        batch_orp2        = [get_image(batch_file, self.image_size, is_crop=self.is_crop, \
                                       resize_w=self.output_size, is_grayscale = 0) \
                             for batch_file in data_orp2]
        self.batch_orp2_images = np.array(batch_orp2).astype(np.float32)


    def __call__(self):
        if self.current_batch >= self.batch_len:
            self.current_batch = 0
            np.random.shuffle(self.bag_ids)

        self.load_new_data()
        self.current_batch += 1            
        return self.batch_r0_images, self.batch_rp1_images, self.batch_rp2_images, self.batch_rn1_images, self.batch_rn2_images, self.batch_or0_images, self.batch_orp1_images, self.batch_orp2_images, self.batch_orn1_images, self.batch_orn2_images

class NoiseSampler(object):
    def __init__(self, args):
        self.sample_size = args.sample_size
        self.code_dim = args.code_dim

    def __call__(self):
        return np.random.normal(loc=0.0, scale=1.0, \
                                        size=(self.sample_size, self.code_dim)).astype(np.float32)
