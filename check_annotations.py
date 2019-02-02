import csv
import os



with open('./images/annotations.csv', mode='r') as csv_file:
    fieldnames = ['path', 'x1', 'y1', 'x2', 'y2', 'class']
    csv_reader = csv.DictReader(csv_file, fieldnames=fieldnames)
    line_count = 0
    img_count = 0
    file_old = ""
    for row in csv_reader:
        line_count = line_count + 1
        file = row["path"].replace('/content/keras-retinanet/', '')
        # file = "."+row["path"]
        if file_old != file:
            file_old = file
            img_count = img_count + 1
            if not os.path.isfile(file):
                print("File does not exist", file)

    print("Number of boxes", line_count )
    print("Number of images", img_count)





