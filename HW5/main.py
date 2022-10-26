import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayDilation(img, kernel):
    temp = img.copy()
    ycenter = int(kernel.shape[0] / 2)
    xcenter = int(kernel.shape[1] / 2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = 0
            for x in range(kernel.shape[0]):
                for y in range(kernel.shape[1]):
                    if kernel[x][y] == 1:
                        xdest = i + x - ycenter
                        ydest = j + y - xcenter
                        if (0 <= xdest < img.shape[0]) and (0 <= ydest < img.shape[0]):
                            pixel = max(pixel, img[xdest][ydest])
            temp[i][j] = pixel
    return temp

def grayErosion(img, kernel):
    temp = img.copy()
    ycenter = int(kernel.shape[0] / 2)
    xcenter = int(kernel.shape[1] / 2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = 255
            for x in range(kernel.shape[0]):
                for y in range(kernel.shape[1]):
                    if kernel[x][y] == 1:
                        xdest = i + x - ycenter
                        ydest = j + y - xcenter
                        if (0 <= xdest < img.shape[0]) and (0 <= ydest < img.shape[0]):
                            pixel = min(pixel, img[xdest][ydest])
            temp[i][j] = pixel
    return temp

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
height, width = img.shape

if not os.path.exists("res"):
    os.mkdir("res")

# Kernel
kernel = np.array([
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
])

# Dilation
dilation = grayDilation(img, kernel)
cv2.imwrite("res/dilation.bmp", dilation)

# Erosion
erosion = grayErosion(img, kernel)
cv2.imwrite("res/erosion.bmp", erosion)

# Opening
opening = grayDilation(erosion, kernel)
cv2.imwrite("res/opening.bmp", opening)

# Closing
closing = grayErosion(dilation, kernel)
cv2.imwrite("res/closing.bmp", closing)