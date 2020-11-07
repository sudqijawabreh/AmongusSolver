import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import pyautogui
import imutils
import importlib

#moduleName = input('O2')
#importlib.import_module(moduleName)

def showImage(title,image):
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def getContoursByLevel(countours,hierarchy,level):
    sameLevel = []
    levels = []
    count = -1 
    clevel = 0
    for hir in sorted(hierarchy[0], key=lambda x: x[3]):
        if hir[1] == -1:
            count += 1
        if count == level:
            clevel = hir[3]

    for con,hir in zip(countours,hierarchy[0]):
        if hir[3] == clevel:
            sameLevel.append(con)
    return sameLevel

def RecogniseDigit(digit):
    coeff = []
    for i in range(0,10):
        if i == 7:
            continue
        template = cv.imread('./' +str(i) + 'o.png',0)
        ret,temp = cv.threshold(template,127,255,cv.THRESH_BINARY)
        invert = (255 - temp)
        coords = cv.findNonZero(invert)
        x, y, w, h = cv.boundingRect(coords)
        rect = temp[y:y+h, x:x+w] 

        if digit.shape[0] < rect.shape[0] or digit.shape[1] < rect.shape[1] :
            #print('no match size resized')
            rect = cv.resize(rect,(digit.shape[1]-1,digit.shape[0]-1))
        #print('image size')
        #print(digit.shape)
        #print('temp size')
        #print(rect.shape)
        w, h = rect.shape[::-1]
        #img = img2.copy()
        meth = 'cv.TM_CCOEFF_NORMED'
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(digit,rect,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        coeff.append((i,max_val,top_left))
    values = (sorted(coeff,key=lambda x: x[1]))
    return values[-1][0]


def GetO2Numbers(image):
    imageName = './o24.png'
    s = cv.imread(imageName)
    img = cv.imread(imageName, 0)
    #s = image.copy()
    #img = s.copy()
    imgArea = img.shape[0] * img.shape[1]
    ret,thresh = cv.threshold(img,200,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    eroded = cv.erode(thresh,kernel,iterations = 1)
    inverted = (255 - eroded) 
    #inverted = (255 - thresh) 
    showImage('image',inverted)
    contours, hierarchy = cv.findContours(inverted, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    count = 0
    areas = []
    for i,cont in zip(range(len(contours)),contours):
        area = cv.contourArea(cont)
        ratio = imgArea / (area+1)
        print(ratio)
        if (ratio > 30  and ratio < 33):
            print('bingo')
            areas.append((i,cont))
    print(areas)
    #cv.drawContours(s, [areas[0][1]], -1, (0,255,0), 3)
    mask = np.zeros_like(thresh)
    cv.drawContours(mask, [areas[0][1]], -1, 255, -1)
    #img1 = cv.imread('o22.png',0)
    ret,big = cv.threshold(img,100,255,cv.THRESH_BINARY)
    showImage('thresh',big)
    out = np.zeros_like(big)
    out[mask == 255] = big[mask == 255]
    kernel = np.ones((1,1),np.uint8)
    anotherImage = cv.erode(out,kernel,iterations = 1)
    #anotherImage = out
    anotherImage = (255 - anotherImage)
    #out= img[mask == 255]
    cv.imwrite("./thresh.png",anotherImage)
    showImage('out',anotherImage)
    contours, hierarchy = cv.findContours(anotherImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    level1 = getContoursByLevel(contours,hierarchy, 2)
    so = sorted(level1, key = lambda x : x[0][0][1])
    so = (so[::-1])[:5] # get only 5 numbers in lower row
    cv.drawContours(s, [contours[2]], -1, (0,255,0), 1)
    #cv.drawContours(s, contours, -1, (0,255,0), 1)
    #cv.drawContours(s, so, -1, (0,255,0), 2)
    mask = np.zeros_like(thresh)
    cv.drawContours(mask, so, -1, 255, -1)
    showImage('original with contour',s)
    o2digits=[]
    for digit in so:
        x,y,w,h = cv.boundingRect(digit)
        roi = thresh[y:y+h,x:x+w]
        out = np.zeros_like(thresh)
        out[mask == 255] = img[mask == 255]
        out = (255 - out)
        #showImage('original with contour',roi)

        print(RecogniseDigit(roi))
        o2digits.append((RecogniseDigit(roi)))
    return o2digits

def GetO2NumKeys(image):
    img = image.copy() 
    ret,thresh = cv.threshold(img,127,255,cv2.THRESH_BINARY)
    meth = 'cv.TM_CCOEFF'
    numLocations = []
    for i in range(0,10):
        template = cv.imread('./' +str(i) + 'n.png',0)
        ret,temp = cv2.threshold(template,127,255,cv2.THRESH_BINARY)
        w, h = temp.shape[::-1]
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img,temp,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        numLocations.append(top_left)
    return numLocations


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

GetO2Numbers('')
