# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:18:42 2020

@author: Biswadeep
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:30:27 2020

@author: Biswadeep
"""

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
#from keras.preprocessing import image
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from numpy import asarray

#The VGGFace architecture
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
from keras.models import model_from_json
model.load_weights('vgg_face_weights.h5')
def extract_face(filename, required_size=(224, 224)):
	# load image from file
	#pixels = plt.imread(filename)
	# create the detector, using default weights

	# detect faces in the image

	# extract the bounding box from the first face

	# extract the face
	face = filename
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

IMAGE_SIZE = [224, 224]
def get_embeddings(filenames):
	# extract faces
	faces = [extract_face(f) for f in filenames]
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples)
	# create a vggface model
	vggface =Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)
	# perform prediction
	yhat = vggface.predict(samples)
	return yhat

filenames = []
path='E:/MachineL/VGGFACE/lulu/Dataset/Sharanya/'
import cv2
import os

def load_images_from_folder(folder):    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            filenames.append(img)
load_images_from_folder(path)



embeddings = get_embeddings(filenames)


import pandas as pd

# Importing the dataabs
pd.DataFrame(embeddings).to_csv("E:/MachineL/VGGFACE/Dataset/Sharanya/database.csv")
dataset = pd.read_csv('E:/MachineL/VGGFACE/Dataset/Sharanya/database.csv')
dataset=dataset.iloc[:, 1:].values