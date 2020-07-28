#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import fashion_mnist
import numpy as np
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt


# In[2]:


(x_train, y_train) , (x_test, y_test) = fashion_mnist.load_data()


# In[3]:


print('Training data shape : ', x_train.shape, y_train.shape)

print('Testing data shape : ', x_test.shape, y_test.shape)


# In[4]:


plt.figure(figsize=(5,5))
plt.subplot(121)
plt.imshow(x_train[0], cmap='gray')
plt.title("Ground Truth :{}".format(y_train[0]))


# In[5]:


x_train = x_train.reshape(-1, 28,28, 1)
x_test = x_test.reshape(-1, 28,28, 1)
x_train.shape, x_test.shape


# In[6]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.
x_test = x_test / 255.


# In[7]:


train_y_one_hot = to_categorical(y_train)
test_y_one_hot = to_categorical(y_test)


# In[8]:


len(test_y_one_hot)


# In[9]:


print('Original label:', y_train[0])
print('After conversion to one-hot:', train_y_one_hot[0])


# In[10]:


from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(x_train, train_y_one_hot, test_size=0.2, random_state=13)


# In[11]:


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


# In[12]:


batch_size = 64
epochs = 20
num_classes = 10


# In[13]:


fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='softmax'))


# In[14]:


fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[15]:


fashion_model.summary()


# In[16]:


fashion_model.fit(train_X, train_label, batch_size=60,epochs=20,verbose=1,validation_data=(valid_X, valid_label))


# In[17]:


fashion_model.save("fashion_model_dropout.h5py")


# In[18]:


test_eval = fashion_model.evaluate(x_test, test_y_one_hot, verbose=1)


# In[19]:


print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


# In[20]:


predicted_classes = fashion_model.predict(x_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape, y_test.shape


# In[21]:


correct = np.where(predicted_classes==y_test)[0]
print ("Found correct labels: ",len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    plt.tight_layout()


# In[ ]:




