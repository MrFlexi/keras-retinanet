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

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
#labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
#labels_to_names = {0: 'person', 1: 'PinBlack', 2: '1x15Blue', 3: '1x9Red', 4: 'JunctionGray', 5: '1x7Red', 6: '1x13Gray', 7: '1x4LRed'}


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
    #print("")
    #print("Boxes", boxes)
    #print("")
    #print("Scores", scores)
    #print("")
    #print("Labels", labels)
    #print("")

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

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


# load image
#print("Hamburg")
#image = read_image_bgr('Hamburg.jpg')
#predict( image )

# Build dictionary with class number and lable name labels_to_names = {0: 'person', 1: 'bicycle')
labels_to_names = {}
with open('./images/classes.csv', mode='r') as csv_file:
    fieldnames = ['label', 'class']
    labels_dict = csv.DictReader(csv_file, fieldnames=fieldnames)

    for row in labels_dict:
        labels_to_names[int(row["class"])] = row["label"]

print(labels_to_names)

print("-----------------------------------------------------------")
print("1x4LRed")
image = read_image_bgr('./images/frame20190125-125636.jpg')
predict( image )

print("-----------------------------------------------------------")
print("1x5LBlack")
image = read_image_bgr('./images/frame20190131-070052.jpg')
predict( image )

print("-----------------------------------------------------------")
print("Multi")
image = read_image_bgr('./images/frame20190131-071501.jpg')
predict( image )