import numpy as np
from keras.models import load_model
import cv2
from PIL import Image

model = load_model('CNNBrainTumorClassifier10Epochs.h5')
print(model.summary())
raw_image = cv2.imread('image.jpg')

imageArray = Image.fromarray(raw_image)
imageProcess = imageArray.resize((64,64))
imgProcess =np.array(imageProcess)
imageProcess  = np.expand_dims(imageProcess ,axis=0)

result = model.predict(imageProcess)
confidence = np.round(result*100,2).max()
print(confidence)
result_final=np.argmax(result,axis=1)
print(result_final)
