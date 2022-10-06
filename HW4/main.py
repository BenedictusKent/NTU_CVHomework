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

def erosionFunc(kernel, original):
    height, width = original.shape
    ycenter = int(kernel.shape[0] / 2)
    xcenter = int(kernel.shape[1] / 2)
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
del dilation

# Erosion
erosion = erosionFunc(kernel, binary)
cv2.imwrite("res/erosion.bmp", erosion)
del erosion