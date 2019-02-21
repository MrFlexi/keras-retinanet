# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import csv

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-f", "--first", required=True,help="first input image")
#ap.add_argument("-s", "--second", required=True,help="second")
#args = vars(ap.parse_args())#


def getBoxes( imageA, imageB ):

	# convert the images to grayscale
	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)


	# compute the Structural Similarity Index (SSIM) between the two
	# images, ensuring that the difference image is returned
	(score, diff) = compare_ssim(grayA, grayB, full=True)
	diff = (diff * 255).astype("uint8")
	print("SSIM: {}".format(score))


	# threshold the difference image, followed by finding contours to
	# obtain the regions of the two input images that differ
	thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	for c in cnts:
		# compute the bounding box of the contour and then draw the
		# bounding box on both input images to represent where the two
		# images differ
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)


	cv2.imshow("Modified", imageB)
	return cnts


def save_boxes_to_csv( boxes ):
	picturePath = '/content/'

	# loop over the contours
	for c in boxes:
		# compute the bounding box of the contour and then draw the
		# bounding box on both input images to represent where the two
		# images differ
		(x, y, w, h) = cv2.boundingRect(c)

		with open('./images/annotations.csv', mode='a', newline='') as csv_file:  # a = append   w = write
			fieldnames = ['path', 'x1', 'y1', 'x2', 'y2', 'class']
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			# writer.writeheader()
			x2 = x + w
			y2 = y + h
			writer.writerow({'path': picturePath, 'x1': x, 'y1': y, 'x2': x2, 'y2': y2, 'class': '1x7Red'})


# load the two input images
imageA = cv2.imread('./webcam/images/background.jpg')
imageB = cv2.imread('./webcam/images/DSC_1334.jpg')

boxes  = getBoxes( imageA, imageB)
save_boxes_to_csv( boxes )
cv2.imshow("Original", imageA)

#cv2.imshow("Diff", diff)
#cv2.imshow("Thresh", thresh)
cv2.waitKey(0)