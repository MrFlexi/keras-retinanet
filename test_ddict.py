import csv

labels_to_names = {}
with open('./images/classes.csv', mode='r') as csv_file:
    fieldnames = ['label', 'class']
    labels_dict = csv.DictReader(csv_file, fieldnames=fieldnames)

    for row in labels_dict:
        print(row["class"], row["label"])
        labels_to_names[int(row["class"])] = row["label"]

print(labels_to_names)