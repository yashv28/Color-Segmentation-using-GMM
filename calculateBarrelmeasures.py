import numpy as np
import cv2
import os
import math

xtrain = np.load('xtrain/xtrain.npy')
xtrain_cov = np.transpose(xtrain)

mean_red = np.mean(xtrain, axis=0)
covariance_matrix_red = np.cov(xtrain_cov)

def calcMeasures(im, orig_img, timg):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    areas = [cv2.contourArea(c) for c in contours]

    heights = []
    widths = []
    dists_average = []
    bottomLeftX = []
    bottomLeftY = []
    topRightY = []
    topRightX = []
    centroidX = []
    centroidY = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if contours.__len__() > 1:
            if np.max(areas)/area <= 4.4:
                flag = 1
            else:
                flag = 0
        else:
            flag = 1
        h = float(h)
        w = float(w)

        if ((h/w) >= 1.15) and ((h/w) <= 2.6): 
            if flag == 1:

                h = int(h)
                w = int(w)
                d_avg = 110.2058805/h + 74.58824/w
                bLeftX = x
                bLeftY = y+h
                tRightX = x+w
                tRightY=y 
                centX=x+w/2.
                centY=y+h/2.

                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(timg, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(orig_img, (int(centX), int(centY)), 1, (0, 255, 0), 2)

                bottomLeftX.append(bLeftX)
                bottomLeftY.append(bLeftY)
                topRightX.append(tRightX)
                topRightY.append(tRightY)
                centroidX.append(centX)
                centroidY.append(centY)

                heights.append(h)
                widths.append(w)
                dists_average.append(d_avg)


    return im, orig_img, timg, dists_average, centroidX, centroidY
    

folder = "2018Proj1_train"
infolder = "EM Output"
final = "Final Output"
i=0
for filename in os.listdir(infolder):
    i=i+1
    basename = os.path.basename(filename)
    print basename
    img = cv2.imread(os.path.join(infolder,filename))
    timg = cv2.imread(os.path.join(folder,filename))
    cv2.imshow('Image', img)
    cv2.waitKey(0)

    im = img
    im = cv2.medianBlur(im, 5)
    morphimg = cv2.morphologyEx(im, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))

    bimg, res, timg, dist, centX, centY = calcMeasures(morphimg, img,timg)

    cv2.imshow('Barrel', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(final, basename[:-7] + 'Final.png'), res)

    print "ImageNo = [0{0}], CentroidX = {1}, CentroidY = {2}, Distance = {3}".format(i,centX,centY,dist)
