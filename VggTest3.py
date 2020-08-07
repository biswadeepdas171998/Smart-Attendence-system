# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:59:40 2020

@author: Biswadeep
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:50:44 2020

@author: Biswadeep
"""

from glob import glob
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
import os
import pandas as pd
import cv2
from scipy.spatial.distance import cosine

folders = glob('Dataset/*')
dataset=pd.read_csv('Label.csv')

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

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    face_array=[]
	# load image from file
    pixels = plt.imread(filename)
	# create the detector, using default weights
    detector = MTCNN()
	# detect faces in the image
    results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
    for i in range(len(results)):
           if results[i]['confidence']>0.99:
	             x1, y1, width, height = results[i]['box']
	             x2, y2 = x1 + width, y1 + height
	             # extract the face
	             face = pixels[y1:y2, x1:x2]
	             # resize pixels to the model size
	             image = Image.fromarray(face)
	             image = image.resize(required_size)
	             face_array.append(asarray(image))
    else:
        return face_array

IMAGE_SIZE = [224, 224]
def get_embeddings(filenames):
	# extract faces
	faces = filenames
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples)
	# create a vggface model
	vggface =Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)
	# perform prediction
	yhat = vggface.predict(samples)
	return yhat

def is_match(known_embedding, candidate_embedding, thresh=0.22):
	# calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    #if score <= thresh:
        #print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    
    return score
        
        #print('>face is a Match (%.3f <= %.3f)' % (score, thresh)
        #print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


faces=extract_face('E:/MachineL/VGGFACE/database/testRealPics/IMG_20200211_144625.jpg')
embeddings=get_embeddings(faces)

match_not_match=[]
for i in range(len(embeddings)):
    match_not_match.append(0)
labels=pd.read_csv('Label.csv')
for i in range(len(folders)):
    for filename in os.listdir(folders[i]):
        csv = pd.read_csv(os.path.join(folders[i],filename))
        csv=csv.iloc[:, 1:].values
        for k in range(len(embeddings)):
            avg=0.0
            for j in range(len(csv)):
                if is_match(csv[j],embeddings[k]) <=0.2:
                    match_not_match[k]=1
                    plt.subplot(121),plt.imshow(faces[k])
                    plt.show()
                    print(labels.iloc[i, 1])
                    break

def new_csv(path,i):
    #newdata=[]
    for filename in os.listdir(path):
        newcsv = pd.read_csv(os.path.join(path,filename))
        newcsv=newcsv.iloc[:, 1:].values
        newfile=np.concatenate((newcsv,np.expand_dims(embeddings[i],axis=0)),axis=0)
        pd.DataFrame(newfile).to_csv(os.path.join(path,filename))
        newdataset = pd.read_csv(os.path.join(path,filename))
        newdataset=newdataset.iloc[:, 1:].values
        #print(newfile)
               
for i in range(len(match_not_match)):
    if match_not_match[i] ==0:
        plt.imshow(faces[i])
        plt.show()
        print('Recognize Unknown Faces')
        for j in range(len(labels)):
            print(str(j)+':'+str(labels.iloc[j, 1]))
        choice = eval(input("Enter Your Choice"))
        if choice >=0 and choice <=13:
            path=folders[choice]
            new_csv(path,i)
        else:
            print('Invalid Choice')