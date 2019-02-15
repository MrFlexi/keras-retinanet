# show images inline
#%matplotlib inline

# automatically reload modules when they have changed
#%load_ext autoreload
#%autoreload 2

# import keras
import keras

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

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases

#model_path = os.path.join('snapshots', 'resnet50_coco_best_v2.1.0.h5')

model_path = os.path.join('snapshots', 'resnet50_lego_01.h5')

#model_path = '/snapshots/resnet50_coco_best_v2.1.0.h5'
# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

def predict( image ):
    # copy to draw on
    draw = image.copy()
    # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

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
        if score < 0.6:
            break

        print("Item  ", labels_to_names[label], score)
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    cv2.imshow('Prediction', draw)

def webcam( ):
    picturePath = './webcam/images/frame4.jpg'
    imageA = cv2.imread('./webcam/images/background.jpg')

    cap = cv2.VideoCapture(0)
    count = 0
    while (True):

        ret, frame = cap.read()
        croppend_image = frame[100:350, 200:450].copy()
        cv2.rectangle(frame, (200, 100), (450, 350), (0, 255, 255), 2)
        cv2.imshow('WebCam', frame)
        predict(croppend_image)
        # Capture frame-by-frame
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# Build dictionary with class number and lable name labels_to_names = {0: 'person', 1: 'bicycle')
labels_to_names = {}
with open('./images/classes.csv', mode='r') as csv_file:
    fieldnames = ['label', 'class']
    labels_dict = csv.DictReader(csv_file, fieldnames=fieldnames)

    for row in labels_dict:
        labels_to_names[int(row["class"])] = row["label"]

print(labels_to_names)
webcam( )

