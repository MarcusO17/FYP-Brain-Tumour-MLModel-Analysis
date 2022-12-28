import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "CombinedDataset")

classes = {"notumor": 0, "glioma": 1, "meningioma": 2,"pituitary": 3}
y=[]
x=[]


for cls in classes:
    path = os.path.join(image_dir,cls)
    for t in os.listdir(path):
        pth = os.path.join(path, t)
        img = cv2.imread(pth)

        x.append(img)
        y.append(classes[cls])

x = np.array(x)
y = np.array(y)
print(x.shape)
x_updated = x.reshape(len(x),-1)
print(x_updated.shape)

x_train, x_test, y_train, y_test = train_test_split(x_updated,y,random_state=10,test_size=.20)

x_train = x_train/255
x_test = x_test/255

classifier = svm.SVC()
classifier.fit(x_train, y_train)

print("training score:",classifier.score(x_train,y_train))
print("test score:",classifier.score(x_test,y_test))


with open("SVMmodelforbraintumor.pickle", "wb") as f:
    pickle.dump(classifier, f)

