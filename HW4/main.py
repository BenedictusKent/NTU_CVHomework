from contextlib import closing
from nis import match
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

######################## Functions ########################
def dilationFunc(kernel, original):
    height, width = original.shape
    ycenter = int(kernel.shape[0] / 2)
    xcenter = int(kernel.shape[1] / 2)
    dilation = original.copy()
    for i in range(height):
        for j in range(width):
            if original[i][j] == 255:
                for x in range(kernel.shape[0]):
                    for y in range(kernel.shape[1]):
                        xdest = i + x - ycenter
                        ydest = j + y - xcenter
                        if (0 <= xdest < height) and (0 <= ydest < width):
                            dilation[xdest][ydest] = 255
    return dilation

def erosionFunc(kernel, original, ycenter, xcenter):
    height, width = original.shape
    erosion = original.copy()
    for i in range(height):
        for j in range(width):
            matchFlag = True
            for x in range(kernel.shape[0]):
                for y in range(kernel.shape[1]):
                    if kernel[x][y] == 1:
                        xdest = i + x - ycenter
                        ydest = j + y - xcenter
                        if (0 <= xdest < height) and (0 <= ydest < width):
                            if original[xdest][ydest] == 0:
                                matchFlag = False
                                break
                        else:
                            matchFlag = False
                            break
            if matchFlag:
                erosion[i][j] = 255
            else:
                erosion[i][j] = 0
    return erosion

def complementFunc(original):
    height, width = original.shape
    comp = original.copy()
    for i in range(height):
        for j in range(width):
            if comp[i][j] == 0:
                comp[i][j] = 255
            else:
                comp[i][j] = 0
    return comp

def intersectFunc(img1, img2):
    height, width = img1.shape
    intersect = img1.copy()
    for i in range(height):
        for j in range(width):
            if img1[i][j] != img2[i][j]:
                intersect[i][j] = 0
    return intersect

###########################################################

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

# Convert to binary image
binary = img.copy()
for i in range(height):
    for j in range(width):
        if binary[i][j] < 128:
            binary[i][j] = 0
        else:
            binary[i][j] = 255
del img

# Dilation
dilation = dilationFunc(kernel, binary)
cv2.imwrite("res/dilation.bmp", dilation)

# Erosion
ycenter = int(kernel.shape[0] / 2)
xcenter = int(kernel.shape[1] / 2)
erosion = erosionFunc(kernel, binary, ycenter, xcenter)
cv2.imwrite("res/erosion.bmp", erosion)

# Opening
opening = dilationFunc(kernel, erosion)
cv2.imwrite("res/opening.bmp", opening)

# Closing
closing = erosionFunc(kernel, dilation, ycenter, xcenter)
cv2.imwrite("res/closing.bmp", closing)

# Hit and miss
kernel = np.array([
    [1, 1],
    [0, 1]
])
erosion1 = erosionFunc(kernel, binary, 1, 0)
comp = complementFunc(binary)
erosion2 = erosionFunc(kernel, comp, 0, 1)
hitmiss = intersectFunc(erosion1, erosion2)
cv2.imwrite("res/hitmiss.bmp", hitmiss)