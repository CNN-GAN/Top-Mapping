import numpy as np
import os, sys
from glob import glob
from scipy.misc import imread, imresize, imsave

def convert_img(file_name):
    img = imread(file_name, mode='RGB')
    img = imresize(img, (224, 224))
    imsave(file_name, img)

for data_id in range(0,3):
    read_path = os.path.join("./GTAV", 'data'+str(data_id), "*.jpg")
    data_file = glob(read_path)
    for file_id in range(len(data_file)):
        print (data_file[file_id])
        convert_img(data_file[file_id])
