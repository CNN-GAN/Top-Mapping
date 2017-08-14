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



class NoiseSampler(object):
    def __init__(self, args):
        self.sample_size = args.sample_size
        self.code_dim = args.code_dim

    def __call__(self):
        return np.random.normal(loc=0.0, scale=1.0, \
                                        size=(self.sample_size, self.code_dim)).astype(np.float32)
