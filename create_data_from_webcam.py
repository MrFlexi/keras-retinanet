import cv2
import os
import time
import keyboard
from skimage.measure import compare_ssim
import argparse
import imutils

def getBoxes( imageA, imageB):

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
	return cnts



def markImage(image,boxes):
    # loop over the contours
    for c in boxes:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return image

def save_boxes_to_csv( cnts ):
    # loop over the contours
    for c in cnts:
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


def extractFrames( ):
    picturePath = './webcam/images/frame4.jpg'
    imageA = cv2.imread('./webcam/images/background.jpg')

    cap = cv2.VideoCapture(0)
    count = 0
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame

        boxes = getBoxes(imageA, frame )
        gray  = markImage(frame, boxes)


        cv2.imshow('frame', gray)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            print('Read %d frame: ' % count, ret)
            cv2.imwrite(os.path.join('./webcam/images/', "frame{:s}.jpg".format(str(time.strftime('%Y%m%d-%H%M%S')))), frame)  # save frame as JPEG file
            count = count + 1

        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():

    extractFrames()

if __name__ == "__main__":
    main()