# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 10:05:37 2020

@author: Gopalji
"""

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential,load_model
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
test_path='D:\dogs-vs-cats\valid\dog'
test_batches=ImageDataGenerator(rescale=1./255).flow_from_directory(test_path,target_size=(64,64),classes=['dog','cat'],batch_size=32,shuffle='False')
model=load_model('dogs_vs_cats1.h5')
print(model.summary())
test_image=image.load_img(r"D:\dogs-vs-cats\valid\cat\cat.10001.jpg",target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
pred=model.predict(test_image)
print(pred[0][1])
plots(test_image)

#model.predict_generator(test_batches,steps=1,verbose=1)
