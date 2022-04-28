import sys
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from pathlib import Path

img_file = sys.argv[1]
Path("out").mkdir(parents=True, exist_ok=True)



def predictor(img_file):
    img = cv2.imread(img_file)
    resize = cv2.resize(img, (64, 64))

    img_fin = np.reshape(resize, [1, 64, 64, 3])
    json_file = open('model/binary.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("model/binary.h5")

    prediction = loaded_model.predict_classes(img_fin)

    prediction = np.squeeze(prediction, axis=1)
    predict = np.squeeze(prediction, axis=0)
    return int(predict)


"""Neural Network Decoding"""
""" The coordinates are created and trained"""
"""-----------------"""
image_width = 300
image_height = 500


def path_file(file):
    return str(file)


def nn(img_file):
    predict = predictor(img_file)
    file = path_file("annotation.csv")
    reader = pd.read_csv(file)

    img = cv2.imread(img_file)
    img = cv2.resize(img, (image_width, image_height))

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)

    fgdModel = np.zeros((1, 65), np.float64)

    rect = (reader.x1[predict], reader.y1[predict], reader.x2[predict], reader.y2[predict])
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, reader.i[predict], cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    img_cut = img*mask2[:, :, np.newaxis]

    cv2.imwrite("out/"+str(img_file), img_cut)


nn(img_file)
