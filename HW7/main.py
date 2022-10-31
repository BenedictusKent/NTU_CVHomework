import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Yokoi Connectivity Number Functions ########################################
def neighborhoodPixels(img, pos):
    pixels = np.zeros(shape=(3,3))
    col, row = pos
    for i in range(3):
        for j in range(3):
            xdest = col + i - 1
            ydest = row + j - 1
            if ((0 <= xdest < img.shape[0]) and (0 <= ydest < img.shape[1])):
                pixels[i][j] = img[xdest][ydest]
            else:
                pixels[i][j] = 0
    return pixels

def hFunc(b, c, d, e):
    if (b == c):
        if ((d != b) or (e != b)):
            return 'q'
        elif ((d == b) and (e == b)):
            return 'r'
    else:
        return 's'

def fFunc(a1, a2, a3, a4):
    if [a1, a2, a3, a4].count('r') == 4:
        return 5
    else:
        return [a1, a2, a3, a4].count('q')

def yokoiNumber(img):
    # result = np.full(img.shape, ' ')
    result = np.zeros(img.shape).astype(int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] != 0:
                pixels = neighborhoodPixels(img, (i,j))
                result[i][j] = fFunc(
                    hFunc(pixels[1][1], pixels[1][2], pixels[0][2], pixels[0][1]),
                    hFunc(pixels[1][1], pixels[0][1], pixels[0][0], pixels[1][0]),
                    hFunc(pixels[1][1], pixels[1][0], pixels[2][0], pixels[2][1]),
                    hFunc(pixels[1][1], pixels[2][1], pixels[2][2], pixels[1][2])
                )
    return result
##############################################################################

# Mark Interior Border #######################################################
def markBorder(source):
    height, width = source.shape
    result = np.full(source.shape, ' ')
    for i in range(height):
        for j in range(width):
            if source[i][j] == 255:
                if (0 <= i < height) and (0 <= j+1 < width) and (source[i][j] == source[i][j+1]):
                    h1 = source[i][j]
                else:
                    h1 = 'b'
                if (0 <= i-1 < height) and (0 <= j < width) and (h1 == source[i-1][j]):
                    h2 = h1
                else:
                    h2 = 'b'
                if (0 <= i < height) and (0 <= j-1 < width) and (h2 == source[i][j-1]):
                    h3 = h2
                else:
                    h3 = 'b'
                if (0 <= i+1 < height) and (0 <= j < width) and (h3 == source[i+1][j]):
                    h4 = h3
                else:
                    h4 = 'b'
                if (h4 != 'b'):
                    f = 'i'
                else:
                    f = 'b'
                result[i][j] = f
    return result
##############################################################################

# Mark Pair Relationship Functions ###########################################
def markPair(border):
    height, width = border.shape
    result = np.full(border.shape, ' ')
    for i in range(height):
        for j in range(width):
            if (border[i][j] != ' '):
                hcount = 0
                if (0 <= j+1 < width) and (border[i][j+1] == 'i'):
                    hcount += 1
                if (0 <= i-1 < height) and (border[i-1][j] == 'i'):
                    hcount += 1
                if (0 <= j-1 < width) and (border[i][j-1] == 'i'):
                    hcount += 1
                if (0 <= i+1 < height) and (border[i+1][j] == 'i'):
                    hcount += 1
                if (hcount < 1) or (border[i][j] != 'b'):
                    result[i][j] = 'q'
                elif (hcount >= 1) and (border[i][j] == 'b'):
                    result[i][j] = 'p'
    return result
##############################################################################

# Thinning ###################################################################
def thin(yokoi, pair, down):
    result = down.copy().astype(int)
    for i in range(yokoi.shape[0]):
        for j in range(yokoi.shape[1]):
            if (yokoi[i][j] == 1) and (pair[i][j] == 'p'):
                result[i][j] = 0
    return result
##############################################################################

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
height, width = img.shape

if not os.path.exists("res"):
    os.mkdir("res")

# Binarize
binarized = img.copy()
for i in range(height):
    for j in range(width):
        if binarized[i][j] < 128:
            binarized[i][j] = 0
        else:
            binarized[i][j] = 255
cv2.imwrite("res/binarized.bmp", binarized)

# Downsample
down = np.zeros(shape=(int(height/8), int(width/8))).astype(int)
for i in range(down.shape[0]):
    for j in range(down.shape[1]):
        down[i][j] = binarized[i*8][j*8]
cv2.imwrite("res/downsampled.bmp", down)

iterations = 1
while True:
    border = markBorder(down)
    pair = markPair(border)
    yokoi = yokoiNumber(down)
    result = thin(yokoi, pair, down)
    # Break if same image, continue otherwise
    if not(np.bitwise_xor(result, down).any()):
        break
    np.savetxt("res/borderInterior_" + str(iterations) + ".txt", border, delimiter='', fmt='%s')
    np.savetxt("res/pairRelationship_" + str(iterations) + ".txt", pair, delimiter='', fmt='%s')
    np.savetxt("res/yokoiNumber_" + str(iterations) + ".txt", yokoi, delimiter='', fmt='%s')
    cv2.imwrite("res/thinning_" + str(iterations) + ".bmp", result)
    down = result
    print("Iteration", iterations)
    iterations += 1