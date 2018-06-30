'''
Script to split edges2handbag AB images into A and B
'''
import cv2
import glob
import os

input_dir = '/home/soonyau/github/gan/pytorch-CycleGAN-and-pix2pix/datasets/edges2handbangs/val/'
output_a_dir = '/home/soonyau/github/gan/MUNIT/datasets/edges2handbags/trainA/'
output_b_dir = '/home/soonyau/github/gan/MUNIT/datasets/edges2handbags/trainB/'

paths = [fname for fname in glob.glob(input_dir+'*')]

for path in paths:
#path = paths[0]

    fname = os.path.split(path)[1].replace('_AB','')
    #print('fname', fname)
    image_ab = cv2.imread(path)
    h, w, c = image_ab.shape
    half_w = int(w/2)

    img_a = image_ab[:,0:half_w,:]
    img_b = image_ab[:,half_w:,:]

    cv2.imwrite(os.path.join(output_a_dir,fname),img_a)
    cv2.imwrite(os.path.join(output_b_dir,fname),img_b)

    '''
    cv2.imshow('a',img_a)
    cv2.imshow('b',img_b)
    cv2.imshow('ab',image_ab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
