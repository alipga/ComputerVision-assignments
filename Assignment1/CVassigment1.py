import cv2 as cv
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


path = './image_set/'

# read all images in the path
for name in glob.glob(path+'*.png'):
    #read mosaic and .png image
    img = cv.imread(name,cv.IMREAD_GRAYSCALE)
    #comment if you want to work in range (0,255)
    img = img.astype(np.float)
    img = img/255
    dest = path+'/result/'+name[len(path):-11]
    original = cv.imread(name[:-11]+'.jpg')
    #comment if want to work in range (0,255)
    original = original.astype(np.float32)
    original = original/255
    #make a mask with the same size as input
    #this mask will represent pattern of mosaic
    # red mosaics are idicated with 0
    mask = np.zeros(img.shape)
    w,h = mask.shape

    for i in range(w):
        for j in range(h):
            # set pixels of green to be 1 and red pixels to be 2
            if  i%2 and not not j%2:
                mask[i,j]=1 #green
            elif not i%2 and not j%2:
                mask[i,j]=2 #blue
            else:
                pass

    # make color mask with mask and input image
    red_mask = (mask == 1).astype(int) * img
    blue_mask = (mask == 0).astype(int) * img
    green_mask = (mask == 2).astype(int) * img

    #kernels for color interpolation
    blue_kernel = np.array([[0,1/4,0],
                            [1/4,1,1/4],
                            [0,1/4,0]],dtype='float')
    red_kernel = np.array([[1/4,1/2,1/4],
                            [1/2,1,1/2],
                            [1/4,1/2,1/4]],dtype='float')
    green_kernel = red_kernel

    # apply kernels to color masks
    red_channel = cv.filter2D(np.float32(red_mask),-1,kernel=red_kernel,borderType=cv.BORDER_CONSTANT)
    blue_channel = cv.filter2D(np.float32(blue_mask),-1,kernel=blue_kernel,borderType=cv.BORDER_CONSTANT)
    green_channel = cv.filter2D(np.float32(green_mask),-1,kernel=green_kernel,borderType=cv.BORDER_CONSTANT)
    print(red_mask[:5,:5])
    print(red_channel[:5,:5])
    # reconstruct image and produce difference
    rec1_img = cv.merge((blue_channel,green_channel,red_channel))
    diff1 = np.sum((rec1_img - original)**2,axis=-1)

    # save results to path/result/
    cv.imwrite(dest+'_rec1.jpg',rec1_img*255)
    cv.imwrite(dest+'_diff1.jpg',diff1*255)
    ####uncomment if want to work in range (0,255)
    # cv.imwrite(dest+'_rec1.jpg',rec1_img)
    # cv.imwrite(dest+'_diff1.jpg',diff1)

    #PART2
    # apply median kernel to green-red and blue-red difference
    GR_diff = cv.medianBlur(green_channel-red_channel,3)
    BR_diff = cv.medianBlur(blue_channel-red_channel,3)

    # reconstruct image and produce difference
    rec2_img = cv.merge((red_channel+BR_diff,red_channel+GR_diff,red_channel))
    print(original.dtype,rec2_img.dtype)
    diff2 = np.sum((original - rec2_img)**2,axis = -1)

    #  save results to path/result/
    cv.imwrite(dest+'_rec2.jpg',rec2_img*255)
    cv.imwrite(dest+'_diff2.jpg',diff2*255)
    ####uncomment if want to work in range (0,255)
    # cv.imwrite(dest+'_rec2.jpg',rec2_img)
    # cv.imwrite(dest+'_diff2.jpg',diff2)
