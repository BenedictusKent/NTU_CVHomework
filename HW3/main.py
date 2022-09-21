import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
height, width = img.shape

if not os.path.exists("res"):
    os.mkdir("res")

# Part A
data = {}
for i in range(height):
    for j in range(width):
        if img[i][j] not in data:
            data[img[i][j]] = 0
        data[img[i][j]] += 1
pixel = list(data.keys())
count = list(data.values())
fig = plt.figure(figsize=(10,5))
plt.bar(pixel, count, color='black')
plt.savefig('res/original_histogram.png')
del data, pixel, count, fig

# Part B
imgdiv3 = []
for i in range(height):
    for j in range(width):
        imgdiv3.append(int(img[i][j] / 3))
imgdiv3 = np.asarray(imgdiv3).reshape(height, width)
data = {}
for i in range(height):
    for j in range(width):
        if imgdiv3[i][j] not in data:
            data[imgdiv3[i][j]] = 0
        data[imgdiv3[i][j]] += 1
pixel = list(data.keys())
count = list(data.values())
fig = plt.figure(figsize=(10,5))
plt.xlim([0, 255])
plt.bar(pixel, count, color='black')
plt.savefig('res/div3_histogram.png')
del imgdiv3, data, pixel, count, fig