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

# Mark Pair Relationship Functions ###########################################
def markPair(img):
    pad = np.zeros((img.shape[0]+2, img.shape[1]+2), dtype=int)
    pad[1:-1, 1:-1] = img
    pair_img = np.zeros(img.shape, dtype=int)
    for i in range(1, pad.shape[0]-1):
        for j in range(1, pad.shape[1]-1):
            if(pad[i][j] != 1):
                pair_img[i-1][j-1] = 0
            elif (pad[i][j+1] == 1 or pad[i-1][j] == 1 or pad[i][j-1] == 1 or pad[i+1][j] == 1):
                pair_img[i-1][j-1] = 1
    return pair_img
##############################################################################

# Thinning ###################################################################
def hShrink(b, c, d, e):
    if (b == c) and (b != d or b != e):
        return 1
    else:
        return 0

def thinning(img, marked):
    pad = np.zeros((img.shape[0]+2, img.shape[1]+2), dtype=int)
    pad[1:-1, 1:-1] = img
    thinned_img = np.zeros(img.shape, dtype=int)
    for i in range(1, pad.shape[0]-1):
        for j in range(1, pad.shape[1]-1):
            if pad[i][j] == 0:
                continue
            a1 = hShrink(pad[i][j], pad[i][j+1], pad[i-1][j+1], pad[i-1][j])
            a2 = hShrink(pad[i][j], pad[i-1][j], pad[i-1][j-1], pad[i][j-1])
            a3 = hShrink(pad[i][j], pad[i][j-1], pad[i+1][j-1], pad[i+1][j])
            a4 = hShrink(pad[i][j], pad[i+1][j], pad[i+1][j+1], pad[i][j+1])
            if ((a1+a2+a3+a4) == 1 and marked[i-1][j-1] == 1):
                thinned_img[i-1][j-1] = 0
                pad[i][j] = 0
            else:
                thinned_img[i-1][j-1] = pad[i][j]
    return thinned_img
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
    yokoi = yokoiNumber(down)
    pair = markPair(yokoi)
    result = thinning(down, pair)
    # Break if same image, continue otherwise
    if not(np.bitwise_xor(result, down).any()):
        break
    np.savetxt("res/pairRelationship_" + str(iterations) + ".txt", pair, delimiter='', fmt='%s')
    np.savetxt("res/yokoiNumber_" + str(iterations) + ".txt", yokoi, delimiter='', fmt='%s')
    cv2.imwrite("res/thinning_" + str(iterations) + ".bmp", result)
    down = result
    print("Iteration", iterations)
    iterations += 1