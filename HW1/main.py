import os
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
height, width = img.shape

##################### Part 1 #####################
if not os.path.exists("res"):
    os.mkdir("res")

# Up-Down Flip
udimg = []
for i in reversed(range(height)):
    for j in range(width):
        udimg.append(img[i][j])
udimg = np.asarray(udimg).reshape(height, width)
cv2.imwrite("res/upside_down.bmp", udimg)
del udimg

# Left-Right Flip
lrimg = []
for i in range(height):
    for j in reversed(range(width)):
        lrimg.append(img[i][j])
lrimg = np.asarray(lrimg).reshape(height, width)
cv2.imwrite("res/righside_left.bmp", lrimg)
del lrimg

# Diagonal Flip
dfimg = np.copy(img)
for i in range(height):
    for j in range(width):
        if i == j:
            break
        else:
            temp = dfimg[i][j]
            dfimg[i][j] = dfimg[j][i]
            dfimg[j][i] = temp
cv2.imwrite("res/diagonal_flip.bmp", dfimg)
del dfimg
##################################################

##################### Part 2 #####################
# Rotate 45 degreees clockwise
rotateimg = ndimage.rotate(img, -45)
rheight, rwidth = rotateimg.shape
for i in range(rheight):
    for j in range(rwidth):
        if rotateimg[i][j] == 0:
            rotateimg[i][j] = 255
cv2.imwrite("res/rotated.bmp", rotateimg)
del rotateimg, rheight, rwidth

# Shrink image
shrink = cv2.resize(img, (int(height/2), int(width/2)), cv2.INTER_AREA)
cv2.imwrite("res/shrinked.bmp", shrink)
del shrink

# Binarize image
_, binarize = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
cv2.imwrite("res/binarize.bmp", binarize)
del binarize
##################################################