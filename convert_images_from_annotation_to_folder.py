import csv
import os
import cv2
import time

main_folder = ".\LegoPredict\lego_fotos\train\"
if not os.path.isdir(main_folder):
    print("Folder does not exist", main_folder)


with open('./images/annotations.csv', mode='r') as csv_file:
    fieldnames = ['path', 'x1', 'y1', 'x2', 'y2', 'class']
    csv_reader = csv.DictReader(csv_file, fieldnames=fieldnames)


    line_count = 0
    img_count = 0
    file_old = ""
    for row in csv_reader:
        line_count = line_count + 1
        target_path = os.path.join(main_folder, row["class"],"frame{:s}.jpg".format(str(time.strftime('%Y%m%d-%H%M%S'))+str(line_count)))
        print("Target",target_path)
        source_file = row["path"].replace('/content/keras-retinanet/', '')
        if os.path.isfile(source_file):
            image= cv2.imread(source_file)
            x1 = int(row["x1"])
            x2 = int(row["x2"])
            y1 = int(row["y1"])
            y2 = int(row["y2"])
            newImage = image[y1:y2, x1:x2].copy()
            cv2.imwrite(target_path, newImage)  # save frame
        else:
            print("File does not exist", source_file)




    print("Number of boxes", line_count )
