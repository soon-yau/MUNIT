import cv2
import numpy as np
import os
from glob import glob

src_dir ="/home/soonyau/dataset/zalando_munit/trainB/"
dest_dir ="/home/soonyau/dataset/zalando_munit/trainA/"

def process_file(src_fname):
    # load file
    print(src_fname)
    src = cv2.imread(src_fname)
    height, width, _ = src.shape
    pad = max((height - width)/2,0)
    #print(width, height, pad)
    src = cv2.copyMakeBorder(src, 0, 0, pad,pad, cv2.BORDER_REPLICATE)
    src = cv2.resize(src, (512, 512))

    # process
    dst = cv2.edgePreservingFilter(src, sigma_s=30, sigma_r=0.8)
    dst = cv2.Canny(dst, 5,6)
    dst = 255 - dst

    # save file
    cv2.imwrite(src_fname, src)

    fname = os.path.basename(src_fname)
    dst_fname = fname.split('.')[0]+'.png'
    cv2.imwrite(os.path.join(dest_dir,dst_fname),dst)

    #cv2.imshow('image', dst)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def process_all():
    src_files = glob(src_dir+"*.jpg")
    for i in range(len(src_files)):
        process_file(src_files[i])


process_all()

#test_file = src_dir+"/004834_1.jpg"
#process_file(test_file)


    


