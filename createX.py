import numpy as np
import os
import cv2

xtrain = []
l=0
for filename in os.listdir("./labeled_data/RedBarrel/"):
    l=l+1
    print l
    basename, extension = os.path.splitext(filename)
    if (extension != ".npy"):
        continue
    redbarrel = np.load('./labeled_data/RedBarrel/'+filename)
    img = cv2.imread('./2018Proj1_train/'+basename+'.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    tmp = img[redbarrel]
    for i in range(tmp.shape[0]):
        xtrain.append(tmp[i, :])

xtrain = np.array(xtrain)

np.save('xtrain', xtrain)