import numpy as np
import os
import plotly.express as px
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import  train_test_split
import cv2
from PIL import Image

model = load_model('CNNBrainTumorClassifier10Epochs.h5')

image_directory = 'datasets2/'
dataset=[]
label=[]

INPUT_SIZE = 64

glioma_images = os.listdir(image_directory+'glioma/')
meningioma_images = os.listdir(image_directory+'meningioma/')
pituitary_images = os.listdir(image_directory+'pituitary/')
notumor_images = os.listdir(image_directory+'notumor/')

for i , image_name in enumerate(glioma_images) :
        if(image_name.split('.')[1] == 'jpg'):
            image = cv2.imread(image_directory+'glioma/'+image_name)
            image = Image.fromarray(image)
            image = image.resize((INPUT_SIZE,INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(0)

for i, image_name in enumerate(pituitary_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'pituitary/' + image_name)
        image = Image.fromarray(image)
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

for i, image_name in enumerate(notumor_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'notumor/' + image_name)
        image = Image.fromarray(image)
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(2)

for i, image_name in enumerate(meningioma_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'meningioma/' + image_name)
        image = Image.fromarray(image)
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(3)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset,label,test_size = 0.2)
print(x_test.shape)

y_pred = model.predict(x_test)
y_pred =np.argmax(y_pred,axis=1)
print(y_test)
print(y_pred)

cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

fig = px.imshow(cm, text_auto=True)
fig.show()