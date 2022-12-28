import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

y=[]
x=[]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "CombinedDataset")

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path))
            img = cv2.imread(path).flatten()
            x.append(img)
            if (label == 'pituitary'):
                y.append(3)
            elif (label == 'glioma'):
                y.append(1)
            elif (label == 'meningioma'):
                y.append(2)
            else:
                y.append(0)

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=10,test_size=0.2)

x_train = x_train/255
x_test = x_test/255

classifier = KMeans(n_clusters=4)
classifier.fit(x_train)

y_pred = classifier.predict(x_test)
print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))

with open("KMeansforbraintumor.pickle", "wb") as f:
    pickle.dump(classifier, f)
