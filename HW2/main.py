import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
height, width = img.shape

##################### Part 1 #####################
if not os.path.exists("res"):
    os.mkdir("res")

# Binary image
binaryimg = []
for i in range(height):
    for j in range(width):
        if img[i][j] < 128:
            binaryimg.append(0)
        else:
            binaryimg.append(255)
binaryimg = np.asarray(binaryimg).reshape(height, width)
cv2.imwrite("res/binary_image.bmp", binaryimg)
del binaryimg

# Histogram
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
plt.savefig('res/histogram.png')
del data, pixel, count, fig