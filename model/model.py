import os
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization, Activation
from skimage import exposure

#Home directory
dir = os.getcwd()
print(dir)

#model Architecture
def deeplab_1(input_shape=(160, 160, 1)):
    inputs = Input(shape=input_shape)

    # Branch 1
    x1 = MaxPooling2D(pool_size=(2, 2))(inputs)
    x1 = Conv2D(64, 3, padding="same", activation="relu")(x1)
    x1 = Conv2D(64, 3, padding="same", activation="relu")(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Conv2D(128, 3, padding="same", activation="relu")(x1)
    x1 = Conv2D(128, 3, padding="same", activation="relu")(x1)
    x1 = MaxPooling2D(pool_size=(1, 1))(x1)
    x1 = Conv2D(256, 3, padding="same", activation="relu")(x1)
    x1 = Conv2D(256, 3, padding="same", activation="relu")(x1)

    # Branch 2
    x2 = Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x2 = Conv2D(64, 3, padding="same", activation="relu")(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Conv2D(128, 3, padding="same", activation="relu")(x2)
    x2 = Conv2D(128, 3, padding="same", activation="relu")(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Conv2D(256, 3, padding="same", activation="relu")(x2)
    x2 = Conv2D(64, 3, padding="same", activation="relu")(x2)

    # Upsampling
    x1 = Conv2DTranspose(256, 3, strides=1, padding="same")(x1)
    x2 = concatenate([x2, x1], axis=-1)
    x2 = Conv2DTranspose(64, 3, strides=2, padding="same")(x2)
    x2 = Conv2DTranspose(1, 3, strides=2, padding="same")(x2)

    outputs = Activation("sigmoid")(x2)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    model.load_weights(str(dir)+"/model/deeplab_model.h5")

    return model


#function which proccess image to enter the model
def img_reshape_to_model(img_path):
    import numpy as np
    from PIL import Image
    img = Image.open(img_path).convert("L")
    h = img.height
    w = img.width
    name_out = os.path.basename(img_path)
    img = img.resize((160, 160))
    img_array = np.asarray(img)
    # Normalize the image
    normalized = np.array(img_array) / 255
    img_array1 = normalized.reshape(1, 160, 160, 1)
    # Apply Histogram equalization
    X = exposure.equalize_hist(img_array1)
    # Apply Contrast stretching
    p2, p98 = np.percentile(X, (2, 98))
    X = exposure.rescale_intensity(X, in_range=(p2, p98))
    
    return X, h, w, name_out

#function which mackes the prediction
def prediction(img_array):
    #model = unet()
    model = deeplab_1()
    #making prediction
    predict = model.predict(img_array)
    # Compute the mean of the array
    mean = np.mean(predict)
    threshold = mean
    thresholded_mask = np.zeros_like(predict)
    thresholded_mask[ predict>= threshold] = 1
    return thresholded_mask

#function which proccess image output of model
def img_reshape_back_from_model(predict1,h , w, name_out):
    predict = predict1.reshape(160, 160)
    predict = cv2.resize(predict,(w, h))
    print(predict)
    img = Image.fromarray(np.uint32(predict*255))
    img = img.convert("L")
    return img

#function which runs pipline of prediction
def pipline(img_path):
    img_array, h, w, name_out = img_reshape_to_model(img_path)
    predict1 = prediction(img_array)
    img = img_reshape_back_from_model(predict1, h , w, name_out )
    print(predict1.shape)
    return img, name_out
