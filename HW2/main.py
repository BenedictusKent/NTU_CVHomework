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

# Connected Components
temp = np.copy(binaryimg)
value = 1
# First pass
for i in range(height):
    for j in range(width):
        if temp[i][j] == 255:
            temp[i][j] = value
            value += 1
# Top down
for i in range(height):
    for j in range(width):
        if temp[i][j] > 0:
            top = -1
            left = -1
            # check top
            if i-1 >= 0:
                if temp[i-1][j] > 0:
                    top = temp[i-1][j]
            # check left
            if j-1 >= 0:
                if temp[i][j-1] > 0:
                    left = temp[i][j-1]
            # check min neighbours
            if top == -1 and left == -1:
                pass
            elif top == -1:
                temp[i][j] = left
            elif left == -1:
                temp[i][j] = top
            else:
                temp[i][j] = min(top, left)
# Bottom up
for i in reversed(range(height)):
    for j in reversed(range(width)):
        if temp[i][j] > 0:
            bottom = -1
            right = -1
            # check bottom
            if i+1 < height:
                if temp[i+1][j] > 0:
                    bottom = temp[i+1][j]
            # check right
            if j+1 < width:
                if temp[i][j+1] > 0:
                    right = temp[i][j+1]
            # check min neighbours
            if bottom == -1 and right == -1:
                pass
            elif bottom == -1:
                temp[i][j] = right
            elif right == -1:
                temp[i][j] = bottom
            else:
                temp[i][j] = min(bottom, right)