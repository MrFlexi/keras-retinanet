import cv2
import os
import time
import keyboard


def extractFrames( ):

    cap = cv2.VideoCapture(0)

    count = 0

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
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