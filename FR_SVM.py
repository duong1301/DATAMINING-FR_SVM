import numpy, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import cv2

path="Dataset/"
data_slice = [70,195,78,172]

resize_ratio = 2.5
h = int((data_slice[1] - data_slice[0])/resize_ratio)
w = int((data_slice[3] - data_slice[2])/resize_ratio)
print("Image dimension after resize: (h, w) = (%d, %d) " % (h, w) )

n_sample = 0
label_count = 0
n_classes = 0
X=[]
Y=[]
target_names = []
for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        img = cv2.imread(path+directory+"/"+file)
        img = cv2.resize(img, (w,h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        featurevector = numpy.array(img).flatten()
        X.append(featurevector)
        Y.append(label_count)
        n_sample = n_sample+1
    target_names.append(directory)
    label_count=label_count+1

n_classes = len(target_names)
print("Sample: ", n_sample)
print("Class: ", target_names)
print("Number of people: ", n_classes)
n_components = 200
pca = PCA(n_components=n_components, whiten=True).fit(X)
eigenfaces = pca.components_.reshape((n_components, h, w))
X_pca = pca.transform(X)
print(X[1])
print(X_pca[1])
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
             'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
clf = RandomizedSearchCV(SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10)
clf = clf.fit(X_pca, Y)
print("Best estimator found by grid search: ")
print(clf.best_estimator_)
test = []
path ="TestSet/data1.jpg"
p = "TestSet/data1.jpg"
testImage = path

testImage=cv2.imread(testImage)
testImage=cv2.resize(testImage, (w,h))
testImage=cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
testImageFeatureVector=numpy.array(testImage).flatten()
test.append(testImageFeatureVector)
testImagePCA = pca.transform(test)
testImagePredict=clf.predict(testImagePCA)
print ("Tên dự đoán : " + target_names[testImagePredict[0]])
img = mpimg.imread(p)
plt.imshow(img)