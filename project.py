#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.datasets import fashion_mnist 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np


# In[2]:


(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()


# In[3]:


train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)


# In[4]:


train_X = train_X.astype('float32')
test_X = test_X.astype('float32')


# In[5]:


train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)


# In[6]:


model = Sequential()

model.add(Conv2D(64, (3,3),activation= "relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(units=10, activation= "softmax"))


# In[7]:


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[8]:


model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=1)


# In[9]:


test_loss, test_acc = model.evaluate(test_X, test_Y_one_hot)


# In[10]:


accuracy = test_acc*100


# In[11]:


acc= open("accuracy.txt","w+")


# In[12]:


acc.write("%d" % accuracy)


# In[13]:


acc.close()


# In[ ]:




