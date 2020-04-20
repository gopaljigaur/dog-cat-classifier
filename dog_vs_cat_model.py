#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
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
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


# In[5]:


train_path='D:/dogs-vs-cats/train'
test_path='D:/dogs-vs-cats/test'
valid_path='D:/dogs-vs-cats/valid'


# In[23]:


train_batches=ImageDataGenerator(rescale=1./255).flow_from_directory(train_path,target_size=(50,50),classes=['dog','cat'],batch_size=32)
valid_batches=ImageDataGenerator(rescale=1./255).flow_from_directory(valid_path,target_size=(50,50),classes=['dog','cat'],batch_size=32)
test_batches=ImageDataGenerator(rescale=1./255).flow_from_directory(test_path,target_size=(50,50),classes=['dog','cat'],batch_size=25,shuffle='False')


# In[20]:


# plots images with labels within jupyter notebook
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


# In[21]:


#imgs,labels=next(train_batches)


# In[22]:


#plots(imgs,titles=labels)


# In[24]:


model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(50,50,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2,activation='sigmoid'))


# In[25]:


model.summary()


# In[26]:


model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[29]:


model.fit_generator(train_batches,steps_per_epoch=700,validation_data=valid_batches,validation_steps=300,epochs=20,verbose=2)


model.save('dogs_vs_cats.h5')
print('model was saved')

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
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
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


# In[5]:


train_path='D:/dogs-vs-cats/train'
test_path='D:/dogs-vs-cats/test'
valid_path='D:/dogs-vs-cats/valid'

train_batches=ImageDataGenerator(rescale=1./255).flow_from_directory(train_path,target_size=(64,64),classes=['dog','cat'],batch_size=32)
valid_batches=ImageDataGenerator(rescale=1./255).flow_from_directory(valid_path,target_size=(64,64),classes=['dog','cat'],batch_size=32)
test_batches=ImageDataGenerator(rescale=1./255).flow_from_directory(test_path,target_size=(50,50),classes=['dog','cat'],batch_size=25,shuffle='False')


model1=Sequential()
model1.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Conv2D(32,(3,3),activation='relu'))
#model1.add(Conv2D(64,(3,3),activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Flatten())
model1.add(Dense(64,activation='relu'))
model1.add(Dense(2,activation='sigmoid'))


# In[25]:


model1.summary()


# In[26]:


model1.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[29]:


model1.fit_generator(train_batches,steps_per_epoch=625,validation_data=valid_batches,validation_steps=157,epochs=15,verbose=2)


model1.save('dogs_vs_cats1.h5')
print('model1 was saved')

from keras.preprocessing import image
from PIL import Image
test_image=image.load_img(r"D:\dogs-vs-cats\test\3.jpg",target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=model1.predict(test_image)
print(result)

train_batches=ImageDataGenerator(rescale=1./255).flow_from_directory(train_path,target_size=(50,50),classes=['dog','cat'],batch_size=40)
valid_batches=ImageDataGenerator(rescale=1./255).flow_from_directory(valid_path,target_size=(50,50),classes=['dog','cat'],batch_size=32)
test_batches=ImageDataGenerator(rescale=1./255).flow_from_directory(test_path,target_size=(50,50),classes=['dog','cat'],batch_size=25,shuffle='False')


model2=Sequential()
model2.add(Conv2D(32,(3,3),input_shape=(50,50,3),activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Conv2D(32,(3,3),activation='relu'))
model2.add(Conv2D(64,(3,3),activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Flatten())
model2.add(Dense(128,activation='relu'))

model2.add(Dense(2,activation='sigmoid'))


# In[25]:


model2.summary()


# In[26]:


model2.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[29]:


model2.fit_generator(train_batches,steps_per_epoch=700,validation_data=valid_batches,validation_steps=300,epochs=20,verbose=2)


model2.save('dogs_vs_cats2.h5')
print('model2 was saved')



train_batches=ImageDataGenerator(rescale=1./255).flow_from_directory(train_path,target_size=(50,50),classes=['dog','cat'],batch_size=32)
valid_batches=ImageDataGenerator(rescale=1./255).flow_from_directory(valid_path,target_size=(50,50),classes=['dog','cat'],batch_size=32)
test_batches=ImageDataGenerator(rescale=1./255).flow_from_directory(test_path,target_size=(50,50),classes=['dog','cat'],batch_size=25,shuffle='False')


model3=Sequential()
model3.add(Conv2D(32,(3,3),input_shape=(50,50,3),activation='relu'))
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Conv2D(32,(3,3),activation='relu'))

model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Flatten())
model3.add(Dense(128,activation='relu'))
model3.add(Dense(2,activation='sigmoid'))


# In[25]:


model3.summary()


# In[26]:


model3.compile(Adam(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])


# In[29]:


model3.fit_generator(train_batches,steps_per_epoch=700,validation_data=valid_batches,validation_steps=300,epochs=20,verbose=2)


model3.save('dogs_vs_cats3.h5')
print('model3 was saved')


# In[30]:
predictions=model.predict_generator(test_batches,steps=1,verbose=2)
for i in predictions:
    print(i)

