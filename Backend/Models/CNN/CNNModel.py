import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense

# Image Storage
image_directory = 'datasets2/'
dataset=[]
label=[]

IMAGE_SIZE = 64

#Image Loading
glioma_images = os.listdir(image_directory+'glioma/')
meningioma_images = os.listdir(image_directory+'meningioma/')
pituitary_images = os.listdir(image_directory+'pituitary/')
notumor_images = os.listdir(image_directory+'notumor/')

for i , image_name in enumerate(glioma_images) :
        if(image_name.split('.')[1] == 'jpg'):
            image = cv2.imread(image_directory+'glioma/'+image_name)
            image = Image.fromarray(image)
            image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
            dataset.append(np.array(image))
            label.append(0)

for i, image_name in enumerate(pituitary_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'pituitary/' + image_name)
        image = Image.fromarray(image)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        dataset.append(np.array(image))
        label.append(1)

for i, image_name in enumerate(notumor_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'notumor/' + image_name)
        image = Image.fromarray(image)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        dataset.append(np.array(image))
        label.append(2)

for i, image_name in enumerate(meningioma_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'meningioma/' + image_name)
        image = Image.fromarray(image)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        dataset.append(np.array(image))
        label.append(3)

#Data Processing
dataset = np.array(dataset)
label = np.array(label)


x_train, x_test, y_train, y_test = train_test_split(dataset,label,test_size = 0.2)

x_train = normalize(x_train)
x_test = normalize(x_test)

#Model Building
model= Sequential()

model.add(Conv2D(32,(3,3), input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation=tf.nn.relu))
model.add(Dense(6,activation=tf.nn.softmax))

#Model Compilation and Saving
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, verbose =1, epochs=10,
          validation_data=(x_test,y_test),shuffle=False)

model.save('CNNBrainTumourClassifier10Epochs')

