import numpy as np
import matplotlib.pyplot as plt
import cv2, os
from roipoly import roipoly
import pdb




############################
## MODIFY THESE VARIABLES ##
############################
inFolder = "2018Proj1_train/"
outFolder = "labeled_data/"
colorClass = "RedBarrel/"
############################


retry = True
rois = []

def on_keypress(event, datafilename, img):
	global retry
	global rois
	if event.key == 'n':
		# save 
		imgSize = np.shape(img)
		mask = np.zeros(imgSize[0:2], dtype=bool)
		for roi in rois:
			mask = np.logical_or(mask, roi.getMask2(imgSize))
		np.save(datafilename, mask)
		print("Saving " + datafilename)
		plt.close()
	elif event.key == 'q':
		print("Quitting")
		exit()
	elif event.key == 'r':
		# retry
		print("Retry annotation")
		rois = []
		retry = True
		plt.close()
	elif event.key == 'a':
		# add
		print("Add another annotation")
		retry = True
		plt.close()

if __name__ == '__main__':
	inFolderPath = inFolder
	outFolderPath = outFolder + colorClass

	for filename in os.listdir(inFolderPath):
		basename, extension = os.path.splitext(filename)
		if (extension != ".png"):
			continue

		textfile = outFolderPath + basename + ".npy"
		if os.path.isfile(textfile):
			continue
		bgrImage = cv2.imread(inFolderPath + filename)
		rgbImg = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)

		rois = []
		retry = True
		while (retry):
			retry = False
			plt.cla()

			# draw region of interest
			plt.imshow(rgbImg, interpolation='none')
			for roi in rois:
				roi.displayROI()
			plt.title(basename)
			rois.append(roipoly(roicolor='r')) #let user draw ROI

			fig = plt.gcf()
			fig.canvas.mpl_connect('key_press_event', \
				lambda event: on_keypress(event, outFolderPath + basename, rgbImg))

			plt.cla()
			plt.imshow(rgbImg, interpolation='none')
			for roi in rois:
				roi.displayROI()
			plt.title("press \'n\' to save and go to next picture, \'r\' to retry \n \'q\' to quit, \'a\' to add another region")
			plt.show()



