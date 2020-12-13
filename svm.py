import numpy as np
import cv2
import glob
import pickle
import skimage
import sklearn
import struct
import matplotlib as plt

# 数据准备
print("读入训练图像和标签...")

f_images = open('train-images.idx3-ubyte','rb')
f_labels = open('train-labels.idx1-ubyte','rb')
f_images_test = open('t10k-images.idx3-ubyte','rb')
f_labels_test = open('t10k-labels.idx1-ubyte','rb')
buf_images = f_images.read()
buf_labels = f_labels.read()
buf_images_test = f_images_test.read()
buf_labels_test = f_labels_test.read()
imgs = []
labels = []
imgs_test = []
labels_test = []

#   images
index = 0
magic, numOfImages, numOfRows, numOfCols = struct.unpack_from('>IIII', buf_images, index)
index += struct.calcsize('>IIII')

for i in range(0,60000):
    img = struct.unpack_from('>784B', buf_images, index)
    index += struct.calcsize('>784B')
    img = np.array(img)
    img = img.reshape(28,28)
    imgs.append(img)

imgs = np.array(imgs)


index = 0
magic_test, numOfImages_test, numOfRows_test, numOfCols_test = struct.unpack_from('>IIII', buf_images_test, index)
index += struct.calcsize('>IIII')

for i in range(0,10000):
    img_test = struct.unpack_from('>784B', buf_images_test, index)
    index += struct.calcsize('>784B')
    img_test = np.array(img_test)
    img_test = img_test.reshape(28,28)
    imgs_test.append(img_test)

imgs_test = np.array(imgs_test)


#   labels
index = 0
magic, numOfLabels = struct.unpack_from('>II', buf_labels, index)
index += struct.calcsize('>II')

for i in range(0,60000):
    label = struct.unpack_from('>1B', buf_labels, index)
    index += struct.calcsize('>1B')
    labels.append(label)

labels = np.array(labels)


index = 0
magic_test, numOfLabels_test = struct.unpack_from('>II', buf_labels_test, index)
index += struct.calcsize('>II')

for i in range(0,10000):
    label_test = struct.unpack_from('>1B', buf_labels_test, index)
    index += struct.calcsize('>1B')
    labels_test.append(label_test)

labels_test = np.array(labels_test)

#特征提取
from skimage.feature import hog
print("提取图像特征...")
X = []
i = 0
for img in imgs:
    i += 1
    fd = hog(img, pixels_per_cell=(4,4), cells_per_block=(2,2), multichannel=False, feature_vector=True)
    X.append(fd)
    if i %10000 == 0:
        print('    已完成%d张图像特征提取'%(i))

X = np.array(X)
Y = np.reshape(np.array(labels),(-1,))

from sklearn import svm

#训练和测试分类器
#   训练
print("学习图像特征...")
clf = svm.LinearSVC(C=1.1)
clf.fit(X,Y)

#   测试
print("测试识别准确率...")
testX = []
testY = []
for img_test in imgs_test:
    fd = hog(img_test, pixels_per_cell=(4,4), cells_per_block=(2,2), multichannel=False, feature_vector=True)
    testX.append(fd)
testX = np.array(testX)
testY = np.reshape(np.array(labels_test),(-1,))

print('HOG: ',clf.score(testX, testY))


