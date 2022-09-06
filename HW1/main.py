from audioop import reverse
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
height, width = img.shape

##################### Part 1 #####################
# Up-Down Flip
udimg = []
for i in reversed(range(height)):
    for j in reversed(range(width)):
        udimg.append(img[i][j])
udimg = np.asarray(udimg).reshape(height, width)
cv2.imwrite("upside_down.bmp", udimg)
del udimg

# Left-Right Flip
lrimg = []
for i in range(height):
    for j in reversed(range(width)):
        lrimg.append(img[i][j])
lrimg = np.asarray(lrimg).reshape(height, width)
cv2.imwrite("righside_left.bmp", udimg)
del lrimg

# Diagonal Flip
dfimg = img
for i in range(height):
    for j in range(width):
        if i == j:
            break
        else:
            dfimg[i][j] = dfimg[j][i]
cv2.imwrite("diagonal_flip.bmp", udimg)
del dfimg
##################################################