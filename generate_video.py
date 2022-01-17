import cv2
import numpy as np
import glob
import os

from functools import cmp_to_key



prefix_path = "/home/fengshikun/Jittor_StyleGan/Color_Symbol1227/samples"
max_resolution = (1302, 782)

img_array = []
file_lst = glob.glob(os.path.join(prefix_path, "*.png"))

file_lst.sort(key = lambda x: (int(os.path.basename(x)[:-4].split("_")[0]), int(os.path.basename(x)[:-4].split("_")[1])))
for filename in file_lst:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    if width < max_resolution[0]:
        img = cv2.resize(img, max_resolution)
    img_array.append(img)


out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, max_resolution)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
