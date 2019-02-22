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

    cnts_filtered = []
    cnts = ""
    if ( score < 0.98 ):
        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        count = 1
        image_height, image_width, image_channels = imageB.shape

        # filter countours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differs
            (x, y, w, h) = cv2.boundingRect(c)

            if ((w * h) > 40000) and ( (x + w) < image_width ):
                cnts_filtered.append(c)


        # draw bounding boxes
        for c in cnts_filtered:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = "Nr. " + str(count) + "  w*h=" + str(w * h)
            print(text)
            cv2.putText(imageB, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            count = count + 1


    return cnts_filtered

def generate_cropped_images(imgOriginal, boxes, csv_path, classname):
    line_count = 0
    main_folder = "cropped_images"
    for c in boxes:
        (x, y, w, h) = cv2.boundingRect(c)
        target_path = os.path.join(main_folder, classname,"frame{:s}.jpg".format(str(time.strftime('%Y%m%d-%H%M%S')) + str(line_count)))
        print(target_path)
        croppend_image = imgOriginal[y:(y+h), x:(x+w)].copy()
        cv2.imwrite(target_path, croppend_image)  # save frame
        line_count = line_count + 1

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
        with open('./images/annotations.csv', mode='a', newline='') as csv_file:  # a = append   w = write
            fieldnames = ['path', 'x1', 'y1', 'x2', 'y2', 'class']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            # writer.writeheader()
            x2 = x + w
            y2 = y + h
            writer.writerow({'path': picturePath, 'x1': x, 'y1': y, 'x2': x2, 'y2': y2, 'class': classname })


def extractFrames( classname ):
    imageA = cv2.imread('./webcam/images/background.jpg')
    path = './webcam/images/'


    count = 0
    for imagename in os.listdir(path):
        img = path + imagename
        print("Image:",img)
        frame = cv2.imread(str(img))
        imgOriginal = frame.copy()
        # Display the resulting frame
        boxes = getBoxes(imageA, frame )
        cv2.putText(frame, imagename, (1, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL);
        cv2.imshow('Image', frame)

        key = cv2.waitKey(0)
        if key & 0xFF == ord('s'):   # s = save image and boxes to annotation file
            print('Read %d frame: ' % count)
            img_path = os.path.join('./images/', "frame{:s}.jpg".format(str(time.strftime('%Y%m%d-%H%M%S'))))
            csv_path = os.path.join('/content/keras-retinanet/images/', "frame{:s}.jpg".format(str(time.strftime('%Y%m%d-%H%M%S'))))
            save_boxes_to_csv( boxes, csv_path , classname )
            cv2.imwrite(img_path, imgOriginal)  # save frame

        if key & 0xFF == ord('c'):  # s = save image and boxes to annotation file

            img_path = os.path.join('./images/', "frame{:s}.jpg".format(str(time.strftime('%Y%m%d-%H%M%S'))))
            csv_path = os.path.join('/content/keras-retinanet/images/')


            generate_cropped_images(imgOriginal, boxes, csv_path, classname)


        if key & 0xFF == ord('q'):
            break
        cv2.destroyAllWindows()



def main():
    classname = "3x5LBlack"
    extractFrames(classname)

if __name__ == "__main__":
    main()