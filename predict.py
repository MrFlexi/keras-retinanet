# show images inline
#%matplotlib inline

# automatically reload modules when they have changed
#%load_ext autoreload
#%autoreload 2

# import keras
import keras
import tkinter as tk
from tkinter import filedialog

#Vor dem Start noch etwas installieren: python setup.py build_ext --inplace

import keras_retinanet

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import csv

root = tk.Tk()
root.withdraw()

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_filenames(root):
    root.withdraw()
    print("Initializing Dialogue... \nPlease select some images.")
    tk_filenames = filedialog.askopenfilenames(initialdir=os.getcwd(), filetypes = [('Images', '.jpg'), ('all files', '*.*'),], title='Please select one or more files')
    filenames = list(tk_filenames)
    return filenames

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)



def predict( image ):
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.7:
            break

        print("Item  ", labels_to_names[label], score)
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()

if __name__ == '__main__':

    print("loading model...")
    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # adjust this to point to your downloaded/trained model
    # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases

    model_path = os.path.join('snapshots', 'resnet50_lego_01.h5')

    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')

    print("start prediction...")
    # Build dictionary with class number and lable name labels_to_names = {0: 'person', 1: 'bicycle')
    labels_to_names = {}
    with open('./images/classes.csv', mode='r') as csv_file:
        fieldnames = ['label', 'class']
        labels_dict = csv.DictReader(csv_file, fieldnames=fieldnames)

        for row in labels_dict:
            labels_to_names[int(row["class"])] = row["label"]

    print(labels_to_names)

    while (True):
        print("get images")
        images = get_filenames(root)

        for image_path in images:
            print("Image", image_path)
            image = read_image_bgr(image_path)
            predict(image)

        print("...prediction ended")

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    print("Ende")
