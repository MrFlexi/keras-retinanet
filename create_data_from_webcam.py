import cv2
import os
import time
from skimage.measure import compare_ssim
import argparse
import imutils
import csv

def getBoxes(imageA: object, imageB: object) -> object:

	# convert the images to grayscale
	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)


	# compute the Structural Similarity Index (SSIM) between the two
	# images, ensuring that the difference image is returned
	(score, diff) = compare_ssim(grayA, grayB, full=True)
	diff = (diff * 255).astype("uint8")
	print("SSIM: {}".format(score))

	cnts = ""
	if ( score < 0.97 ):
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
	return cnts


def save_boxes_to_csv( cnts, path, classname ):
	#picturePath = '/content/'
	picturePath = path
	#loop over the contours
	for c in cnts:
		# compute the bounding box of the contour and then draw the
		# bounding box on both input images to represent where the two
		# images differ
		# Only store boxes with a width > 20 pixel
		(x, y, w, h) = cv2.boundingRect(c)
		if ( w > 20 ):
			with open('./images/annotations.csv', mode='a', newline='') as csv_file:  # a = append   w = write
				fieldnames = ['path', 'x1', 'y1', 'x2', 'y2', 'class']
				writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
				# writer.writeheader()
				x2 = x + w
				y2 = y + h
				writer.writerow({'path': picturePath, 'x1': x, 'y1': y, 'x2': x2, 'y2': y2, 'class': classname })


def extractFrames( classname ):
	picturePath = './webcam/images/frame4.jpg'
	imageA = cv2.imread('./webcam/images/background.jpg')

	cap = cv2.VideoCapture(0)
	count = 0
	while (True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		imgOriginal = frame.copy()

		# Display the resulting frame
		boxes = getBoxes(imageA, frame )
		cv2.imshow('frame', frame)

		key = cv2.waitKey(1)
		if key & 0xFF == ord('s'):   # s = save image and boxes to annotation file
			print('Read %d frame: ' % count, ret)
			img_path = os.path.join('./images/', "frame{:s}.jpg".format(str(time.strftime('%Y%m%d-%H%M%S'))))
			csv_path = os.path.join('/content/keras-retinanet/images/', "frame{:s}.jpg".format(str(time.strftime('%Y%m%d-%H%M%S'))))
			save_boxes_to_csv( boxes, csv_path , classname )
			cv2.imwrite(img_path, imgOriginal)  # save frame

		if key & 0xFF == ord('b'):   # b = sve new background image
			print('Read %d frame: ' % count, ret)
			path = os.path.join('./webcam/images/', "background.jpg".format(str(time.strftime('%Y%m%d-%H%M%S'))))
			save_boxes_to_csv( boxes, path , classname )
			cv2.imwrite(path, imgOriginal)  # save frame

		if key & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

def main():
	classname = "Pin3Blue"
	extractFrames(classname)

if __name__ == "__main__":
	main()