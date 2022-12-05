import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def padding(source, pad):
    return cv2.copyMakeBorder(source, top=pad, bottom=pad, left=pad, right=pad, borderType=cv2.BORDER_REPLICATE)

def zeroCrossing(mask):
    padded = padding(mask, 1)
    result = np.zeros(mask.shape, dtype=int)
    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            if padded[i][j] != 1:
                result[i-1][j-1] = 255
            else:
                cross = 1
                for x in range(3):
                    if cross == 1:
                        for y in range(3):
                            xdest = i + x - 1
                            ydest = j + y - 1
                            if padded[xdest][ydest] == -1:
                                cross = 0
                if cross == 1:
                    result[i-1][j-1] = 255
                else:
                    result[i-1][j-1] = 0
    return result

def laplacianMask(source, kernel, threshold):
    padded = padding(source, 1)
    result = np.zeros(source.shape, dtype=int)
    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            # Region of interest
            box = []
            for x in range(3):
                for y in range(3):
                    xdest = i + x - 1
                    ydest = j + y - 1
                    box.append(padded[xdest][ydest])
            # Multiply RoI with kernel
            total = 0
            for x in range(len(kernel)):
                total += (kernel[x] * box[x])
            total = round(total, 2)     # to avoid floating point precision error
            # Insert to mask
            if total >= threshold:
                result[i-1][j-1] = 1
            elif total <= -threshold:
                result[i-1][j-1] = -1
            else:
                result[i-1][j-1] = 0
    return result

def problemABC(source, kernel, threshold):
    mask = laplacianMask(source, kernel, threshold)
    result = zeroCrossing(mask)
    return result

def lgMask(source, kernel, threshold):
    padded = padding(source, 5)
    result = np.zeros(source.shape, dtype=int)
    for i in range(5, padded.shape[0]-5):
        for j in range(5, padded.shape[1]-5):
            # Region of interest
            box = []
            for x in range(11):
                for y in range(11):
                    xdest = i + x - 5
                    ydest = j + y - 5
                    box.append(padded[xdest][ydest])
            # Multiply RoI with kernel
            total = 0
            index = 0
            for x in range(11):
                for y in range(11):
                    total += (kernel[x][y] * box[index])
                    index += 1
            # Insert to mask
            if total >= threshold:
                result[i-5][j-5] = 1
            elif total <= -threshold:
                result[i-5][j-5] = -1
            else:
                result[i-5][j-5] = 0
    return result

def problemDE(source, kernel, threshold):
    mask = lgMask(source, kernel, threshold)
    result = zeroCrossing(mask)
    return result

if __name__ == '__main__':
    img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    if not os.path.exists("res"):
        os.mkdir("res")

    kernel = [0, 1, 0, 1, -4, 1, 0, 1, 0]
    result = problemABC(img, kernel, 15)
    cv2.imwrite("res/laplacian_1.png", result)

    kernel = [1, 1, 1, 1, -8, 1, 1, 1, 1]
    kernel = [x / 3 for x in kernel]
    result = problemABC(img, kernel, 15)
    cv2.imwrite("res/laplacian_2.png", result)
    
    kernel = [2, -1, 2, -1, -4, -1, 2, -1, 2]
    kernel = [x / 3 for x in kernel]
    result = problemABC(img, kernel, 20)
    cv2.imwrite("res/minimum_variance_laplacian.png", result)

    kernel = [
        [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
        [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
        [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
        [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
        [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
        [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
        [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
        [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
        [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
        [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
        [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]
    ]
    result = problemDE(img, kernel, 3000)
    cv2.imwrite("res/laplacian_of_gaussian.png", result)

    kernel = [
        [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
        [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
        [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
        [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
        [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
    ]
    result = problemDE(img, kernel, 1)
    cv2.imwrite("res/difference_of_gaussian.png", result)
