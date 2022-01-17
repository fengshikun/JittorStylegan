import cv2

import glob
import argparse
import os
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare the training data')
    
    parser.add_argument('--max_size', type=int, default=64, help='max size of the result image')
    parser.add_argument('--start_size', type=int, default=8, help='minimal size of the result image')
    parser.add_argument('root_path', type=str, help='root path of the img file')

    args = parser.parse_args()
    suffix = "*.png"
    file_lst = glob.glob(args.root_path + "/{}".format(suffix))
    if not len(file_lst):
        file_lst = glob.glob(args.root_path + "/{}".format("*.jpg"))
    
    base_path = os.path.dirname(args.root_path)

    start_size = args.start_size
    while start_size <= args.max_size:
        write_path = os.path.join(base_path, str(start_size))
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        for file in file_lst:
            img = cv2.imread(file)
            r_img = cv2.resize(img, (start_size, start_size), interpolation=cv2.INTER_LANCZOS4)
            img_name = os.path.basename(img)
            cv2.imwrite(os.path.join(write_path, img_name), r_img)

            