import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def padding(source, pad):
    return cv2.copyMakeBorder(source, top=pad, bottom=pad, left=pad, right=pad, borderType=cv2.BORDER_REPLICATE)

def robert(source, threshold):
    result = np.zeros(source.shape, dtype=int)
    height, width = result.shape
    for i in range(height):
        for j in range(width):
            box = []
            for x in range(2):
                for y in range(2):
                    xdest = i + x
                    ydest = j + y
                    if (xdest < height) and (ydest < width):
                        box.append(source[xdest][ydest])
                    else:
                        box.append(source[i][j])
            r1 = -int(box[0]) + int(box[3])
            r2 = -int(box[1]) + int(box[2])
            gradient = ((r1**2) + (r2**2)) ** 0.5
            if gradient < threshold:
                result[i][j] = 255
    return result

def prewitt(source, threshold):
    padded = padding(source, 1)
    result = np.zeros(source.shape, dtype=int)
    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            box = []
            for x in range(3):
                for y in range(3):
                    xdest = i + x - 1
                    ydest = j + y - 1
                    box.append(padded[xdest][ydest])
            p1 = -int(box[0])-int(box[1])-int(box[2])+int(box[6])+int(box[7])+int(box[8])
            p2 = -int(box[0])-int(box[3])-int(box[6])+int(box[2])+int(box[5])+int(box[8])
            gradient = ((p1**2) + (p2**2)) ** 0.5
            if gradient < threshold:
                result[i-1][j-1] = 255
    return result

def sobel(source, threshold):
    padded = padding(source, 1)
    result = np.zeros(source.shape, dtype=int)
    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            box = []
            for x in range(3):
                for y in range(3):
                    xdest = i + x - 1
                    ydest = j + y - 1
                    box.append(padded[xdest][ydest])
            p1 = -int(box[0])-(int(box[1]) * 2)-int(box[2])+int(box[6])+(int(box[7]) * 2)+int(box[8])
            p2 = -int(box[0])-(int(box[3]) * 2)-int(box[6])+int(box[2])+(int(box[5]) * 2)+int(box[8])
            gradient = ((p1**2) + (p2**2)) ** 0.5
            if gradient < threshold:
                result[i-1][j-1] = 255
    return result

def freiandchen(source, threshold):
    padded = padding(source, 1)
    result = np.zeros(source.shape, dtype=int)
    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            box = []
            for x in range(3):
                for y in range(3):
                    xdest = i + x - 1
                    ydest = j + y - 1
                    box.append(padded[xdest][ydest])
            p1 = -int(box[0])-(int(box[1]) * (2**0.5))-int(box[2])+int(box[6])+(int(box[7]) * (2**0.5))+int(box[8])
            p2 = -int(box[0])-(int(box[3]) * (2**0.5))-int(box[6])+int(box[2])+(int(box[5]) * (2**0.5))+int(box[8])
            gradient = ((p1**2) + (p2**2)) ** 0.5
            if gradient < threshold:
                result[i-1][j-1] = 255
    return result

def kirsch(source, threshold):
    padded = padding(source, 1)
    result = np.zeros(source.shape, dtype=int)
    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            box = []
            for x in range(3):
                for y in range(3):
                    xdest = i + x - 1
                    ydest = j + y - 1
                    box.append(padded[xdest][ydest])
            k0 = (-3*(int(box[1])+int(box[0])+int(box[3])+int(box[6]+int(box[7])))) + (5*(int(box[2])+int(box[5])+int(box[8])))
            k1 = (-3*(int(box[8])+int(box[0])+int(box[3])+int(box[6]+int(box[7])))) + (5*(int(box[1])+int(box[2])+int(box[5])))
            k2 = (-3*(int(box[8])+int(box[5])+int(box[3])+int(box[6]+int(box[7])))) + (5*(int(box[0])+int(box[1])+int(box[2])))
            k3 = (-3*(int(box[8])+int(box[5])+int(box[2])+int(box[6]+int(box[7])))) + (5*(int(box[0])+int(box[1])+int(box[3])))
            k4 = (-3*(int(box[8])+int(box[5])+int(box[2])+int(box[1]+int(box[7])))) + (5*(int(box[0])+int(box[3])+int(box[6])))
            k5 = (-3*(int(box[8])+int(box[5])+int(box[2])+int(box[1]+int(box[0])))) + (5*(int(box[3])+int(box[6])+int(box[7])))
            k6 = (-3*(int(box[3])+int(box[5])+int(box[2])+int(box[1]+int(box[0])))) + (5*(int(box[6])+int(box[7])+int(box[8])))
            k7 = (-3*(int(box[3])+int(box[6])+int(box[2])+int(box[1]+int(box[0])))) + (5*(int(box[5])+int(box[7])+int(box[8])))
            gradient = max(k0,k1, k2, k3, k4, k5, k6, k7)
            if gradient < threshold:
                result[i-1][j-1] = 255
    return result

def robinson(source, threshold):
    padded = padding(source, 1)
    result = np.zeros(source.shape, dtype=int)
    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            box = []
            for x in range(3):
                for y in range(3):
                    xdest = i + x - 1
                    ydest = j + y - 1
                    box.append(padded[xdest][ydest])
            r0 = -int(box[0])-(int(box[3]) * 2)-int(box[6])+int(box[2])+(int(box[5]) * 2)+int(box[8])
            r1 = -int(box[3])-(int(box[6]) * 2)-int(box[7])+int(box[1])+(int(box[2]) * 2)+int(box[5])
            r2 = -int(box[6])-(int(box[7]) * 2)-int(box[8])+int(box[0])+(int(box[1]) * 2)+int(box[2])
            r3 = -int(box[5])-(int(box[8]) * 2)-int(box[7])+int(box[1])+(int(box[0]) * 2)+int(box[3])
            r4 = -r0
            r5 = -r1
            r6 = -r2
            r7 = -r3
            gradient = max(r0, r1, r2, r3, r4, r5, r6, r7)
            if gradient < threshold:
                result[i-1][j-1] = 255
    return result

def nevatiababu(source, threshold):
    padded = padding(source, 2)
    result = np.zeros(source.shape, dtype=int)
    for i in range(2, padded.shape[0]-2):
        for j in range(2, padded.shape[1]-2):
            box = []
            for x in range(5):
                for y in range(5):
                    xdest = i + x - 2
                    ydest = j + y - 2
                    box.append(padded[xdest][ydest])
            n0 = 100 * (int(box[0]) + int(box[1]) + int(box[2]) + int(box[3]) + int(box[4]) + int(box[5]) + int(box[6]) + int(box[7]) + int(box[8]) + int(box[9]))
            n0 += -100 * (int(box[15]) + int(box[16]) + int(box[17]) + int(box[18]) + int(box[19]) + int(box[20]) + int(box[21]) + int(box[22]) + int(box[23]) + int(box[24]))
            
            n1 = 100 * (int(box[0]) + int(box[1]) + int(box[2]) + int(box[3]) + int(box[4]) + int(box[5]) + int(box[6]) + int(box[7])) + (78*int(box[8])) + (-32*int(box[9])) + (100*int(box[10])) + (92*int(box[11]))
            n1 += (-92*int(box[13])) - (100*int(box[14])) + (32*int(box[15])) - (78*int(box[16])) + (-100*(int(box[17])+int(box[18])+int(box[19])+int(box[20])+int(box[21])+int(box[22])+int(box[23])+int(box[24])))
            
            n2 = 100 * (int(box[0]) + int(box[1]) + int(box[2]) + int(box[5]) + int(box[6]) + int(box[10]) + int(box[11]) + int(box[15]) + int(box[20])) - (78*int(box[8])) + (32*int(box[3])) + (92*int(box[7]))
            n2 += (-92*int(box[17])) - (32*int(box[21])) + (78*int(box[16])) + (-100*(int(box[4])+int(box[9])+int(box[13])+int(box[14])+int(box[18])+int(box[19])+int(box[22])+int(box[23])+int(box[24])))

            n3 = -100 * (int(box[0]) + int(box[1]) + int(box[5]) + int(box[6]) + int(box[10]) + int(box[11]) + int(box[15]) + int(box[16]) + int(box[20]) + int(box[21]))
            n3 += 100 * (int(box[3]) + int(box[4]) + int(box[8]) + int(box[9]) + int(box[13]) + int(box[14]) + int(box[18]) + int(box[19]) + int(box[23]) + int(box[24]))

            n4 = -100 * (int(box[0]) + int(box[5]) + int(box[10]) + int(box[11]) + int(box[15]) + int(box[16]) + int(box[20]) + int(box[21]) + int(box[22])) - (78*int(box[6])) + (32*int(box[2])) + (92*int(box[7]))
            n4 += (-92*int(box[17])) - (32*int(box[23])) + (78*int(box[18])) + (100*(int(box[2])+int(box[3])+int(box[4])+int(box[8])+int(box[9])+int(box[13])+int(box[14])+int(box[19])+int(box[24])))

            n5 = 100 * (int(box[0]) + int(box[1]) + int(box[2]) + int(box[3]) + int(box[4]) + int(box[7]) + int(box[8]) + int(box[9]) + int(box[14])) + (78*int(box[6])) - (32*int(box[5])) - (92*int(box[11]))
            n5 += (92*int(box[13])) + (32*int(box[19])) - (78*int(box[18])) - (100*(int(box[10])+int(box[15])+int(box[16])+int(box[17])+int(box[20])+int(box[21])+int(box[22])+int(box[23])+int(box[24])))

            gradient = max(n0, n1, n2, n3, n4, n5)
            if gradient < threshold:
                result[i-2][j-2] = 255
    return result

if __name__ == '__main__':
    img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    if not os.path.exists("res"):
        os.mkdir("res")

    result = robert(img, 12)
    cv2.imwrite("res/robert.png", result)

    result = prewitt(img, 24)
    cv2.imwrite("res/prewitt.png", result)

    result = sobel(img, 38)
    cv2.imwrite("res/sobel.png", result)

    result = freiandchen(img, 30)
    cv2.imwrite("res/freiandchen.png", result)

    result = kirsch(img, 135)
    cv2.imwrite("res/kirsch.png", result)

    result = robinson(img, 43)
    cv2.imwrite("res/robinson.png", result)

    result = nevatiababu(img, 12500)
    cv2.imwrite("res/nevatiababu.png", result)