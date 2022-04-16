import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import pyautogui
import imutils
import importlib
import math
import pdb
debug = True
#moduleName = input('O2')
#importlib.import_module(moduleName)
def show(image):
    cv.imshow('image',image)
    cv.waitKey(0)
    cv.destroyAllWindows()


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
        template = cv.imread('./images/' +str(i) + 'o.png',0)
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

def GetBoxCenter(box):
        x,y,w,h= box
        center = [(x + (w//2)),(y+(h//2))]
        return center

def GetLowerLine(contours,diviation,imgSize):
    centers = []

    #for c1 in contours:
    #    box1 = cv.boundingRect(c1)
    #    print(GetBoxCenter(box1))
    #    print(box1)
    #    box1 = rotate_box(box1,25)
    #    center = GetBoxCenter(box1)
    #    centers.append(center)
    #print(centers)
    #return np.array(centers,dtype=np.int32).reshape((-1,1,2))
    
    lines = set() 
    for c1 in contours:
        box1 = cv.boundingRect(c1)
        elements = set()
        cel = []
        box1 = rotate_box(box1,25,imgSize)
        line = GetBoxCenter(box1)
        sumC =  0
        for c2 in contours:
            box2 = cv.boundingRect(c2)
            box2 = rotate_box(box2,25,imgSize)
            center = GetBoxCenter(box2)
            if abs(line[1] - center[1]) <= diviation:
                elements.add(box2)
                cel.append(c2)
                sumC += center[1]

        if elements not in lines:
            avg = sumC / (len(centers) + 1)
            lines.add(frozenset(elements))
            centers.append((avg,cel))

    centers = sorted(centers, key=lambda x: x[0], reverse = True)
    centers = list(map(lambda x: list(x[1]), list(centers)))
    #print(list(map(lambda x: len(x),centers)))

    #breakpoint()
    return centers


def GetO2Numbers(image):
    #imageName = './images/hello.png'
    #imageName = './images/o25.png'
    #s = cv.imread(imageName)
    #img = cv.cvtColor(s,cv.COLOR_BGR2GRAY)
    #img = cv.imread(imageName, 0)
    s = image.copy()
    if isRedOverlay(s):
        s = RemoveRedOverlay(s)
        #show(s)
    img = cv.cvtColor(s,cv.COLOR_BGR2GRAY)
    imgArea = img.shape[0] * img.shape[1]
    ret,thresh = cv.threshold(img,195,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    eroded = cv.erode(thresh,kernel,iterations = 1)
    inverted = (255 - eroded) 
    #inverted = (255 - thresh) 
    if debug:
        showImage('image',inverted)
    contours, hierarchy = cv.findContours(inverted, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    count = 0
    areas = []
    for i,cont in zip(range(len(contours)),contours):
        area = cv.contourArea(cont)
        ratio = imgArea / (area+1)
        #print(ratio)
        if (ratio > 30  and ratio < 33):
            #print('bingo')
            areas.append((i,cont))
    #cv.drawContours(s, [areas[0][1]], -1, (0,255,0), 3)
    mask = np.zeros_like(thresh)
    cv.drawContours(mask, [areas[0][1]], -1, 255, -1)
    #img1 = cv.imread('o22.png',0)
    ret,big = cv.threshold(img,97,255,cv.THRESH_BINARY)
    if debug:
        showImage('thresh',big)
    out = np.zeros_like(big)
    out[mask == 255] = big[mask == 255]
    #kernel = np.ones((1,1),np.uint8)
    kernel = cv.getStructuringElement(cv2.MORPH_CROSS,(1,1))
    anotherImage = cv.dilate(out,kernel,iterations = 1)
    anotherImage = cv.erode(anotherImage,kernel,iterations = 1)
    #anotherImage = out
    anotherImage = (255 - anotherImage)
    #out= img[mask == 255]
    if debug:
        showImage('out',anotherImage)
    contours, hierarchy = cv.findContours(anotherImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    level1 = getContoursByLevel(contours,hierarchy, 2)
    print(img.shape)
    centers = (GetLowerLine(level1,5,(img.shape[0],img.shape[1])))[0]
    sortedCenters = sorted(centers, key = lambda x: cv.contourArea(x), reverse = True)
    only5 = sorted(sortedCenters[:5], key = lambda x:cv.boundingRect(x)[0])
    
    othercenters = []
    #othercenters = (GetLowerLine(other))
    #cv.drawContours(s, [contours[2]], -1, (0,255,0), 1)
    #cv.drawContours(s, centers[0], -1, (0,255,0), 1)
    cv.drawContours(s, only5, -1, (0,255,0), 2)
    #cv.polylines(s,[centers],False,(0,255,0),1)
    mask = np.zeros_like(thresh)
    cv.drawContours(mask, only5, -1, 255, -1)
    if debug:
        showImage('original with contour',s)
    o2digits=[]
    for digit in only5:
        x,y,w,h = cv.boundingRect(digit)
        roi = thresh[y:y+h,x:x+w]
        out = np.zeros_like(thresh)
        out[mask == 255] = img[mask == 255]
        out = (255 - out)
        if debug:
            showImage('original with contour',roi)

        if debug:
            print(RecogniseDigit(roi))
        o2digits.append((RecogniseDigit(roi)))
    return o2digits

def GetO2NumKeys(image):
    #img = imread('./images/hello.png',0)
    img = image.copy() 
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(img,127,255,cv2.THRESH_BINARY)
    meth = 'cv.TM_CCOEFF'
    numLocations = []
    files = ['./images/' + str(i) + 'n.png' for i in range (10)]
    files.append('tick.png')
    for file in files:
        template = cv.imread(file, 0)
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

# rotate anlge degrees around center value
def rotate_point(point, angle,center):
    x, y = point
    cx , cy = center
    x -= cx
    y -= cy
    xprime = x * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle))
    yprime = y * math.cos(math.radians(angle)) + x * math.sin(math.radians(angle))
    xprime += cx
    yprime += cy
    return (xprime, yprime)

# rotate around image center
def rotate_box(box, angle, imgSize):
    x,y,w,h=box
    #center = GetBoxCenter(box) rotate around box center
    center = (imgSize[0]/2,imgSize[1]/2)
    xprime , yprime = rotate_point((x,y),angle,center)
    return  (xprime,yprime,w,h)

def RemoveRedOverlay(image):
    #making overlay follows this equation
    # p = alpha*255 + beta*value where beta = 1 - alpah
    #reverse it as value = (p - (alpha*255)) / beta
    alpha = 0.3719806763285024 # solved for alpha from examples red.png and nored.png
    beta = 1 - alpha
    red = np.zeros_like(image)
    red[:,:,2] = 255
    original = (image - (alpha*red)) / beta
    return original.astype(np.uint8)

def isRedOverlay(img):
    return np.all(img[:,:,2] > 45)


def GetEstimation(contrs):
    approxes = []
    for cnt in contrs:
        epsilon = 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        if len(approx) <= 4:
            approxes.append(approx)
    return approxes

def test(img):
    img1 = cv.cvtColor(img.copy(),cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img1,100,200)
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    wRatio = img1.shape[1]/1366
    hRatio = img.shape[0]/768
    ih = 701 - 67
    iw = 1000 -  495
    x = int(495 * wRatio) +5
    y = int(67 * hRatio) +5
    h = int(ih * hRatio) - 7
    w = int(iw * wRatio) -7


    edges = cv.resize(edges,(edges.shape[1]*2,edges.shape[0]*2))
    kernel = np.ones((3,3),np.uint8)
    erosion = cv.dilate(edges,kernel,iterations = 2)
    erosion = cv.erode(erosion,kernel,iterations = 1)
    erosion = cv.resize(erosion,(erosion.shape[1]//2,erosion.shape[0]//2))
    #cv.imshow('output',erosion)
    #cv.waitKey(0)
    edges = erosion

    #approxes = GetEstimation(contours)
    #maxArea= -1
    #largestBox = []
    #sortedApproxes = sorted(contours,key=lambda x: cv.contourArea(x), reverse=True)[:50]
    #boxWithLeafes = sortedApproxes[0]
    #boxToEnter = sortedApproxes[1]
    #cv.drawContours(img, sortedApproxes, -1, (0,255,0), 2)
    #show(img)
    #x,y,w,h = cv.boundingRect(boxWithLeafes)
    firstRect = edges[y:y+h, x:x+w] 
    the = img1[y:y+h, x:x+w] 
    ret,temp = cv.threshold(the,120,255,cv.THRESH_BINARY)
    temp = (255 - temp)
    kernel = np.ones((7,7),np.uint8)
    for i in range(9):
        temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
    #output = cv.copyMakeBorder(closing, 0, 0, 0, 0, cv.BORDER_CONSTANT,value=0)
    #show(output)

    rect = temp
    orgRect = img[y:y+h, x:x+w]
    contours, hierarchy = cv.findContours(rect, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    level2 = sorted((contours),key=lambda x: cv.contourArea(x), reverse=True)[:6]
    rectArea = x * h
    level2 = list(filter(lambda x : cv.contourArea(x) < 0.5 * rectArea, level2))
    cv.drawContours(orgRect, level2, -1, (0,255,0), 2)
    if debug :
        show(rect)
        show(orgRect)
    if (len(level2) <= 1):
        return []
    if debug:
        boundingRects = [cv.boundingRect(contr) for contr in level2][:6]
        for boundRect in boundingRects:
            cv.rectangle(orgRect,boundRect,(0,0,255),1)
        #centers = [GetBoxCenter(box) for box in boundingRects]
        #centers = [(x1+x,y1+y) for (x1,y1) in centers]
        show(edges)
        #allImageRect = [(x1+x,y1+y,w1,h1)for (x1,y1,w1,h1) in boundingRects]
        #for boundRect in allImageRect:
        #    cv.rectangle(img,boundRect,(0,0,255),1)
    toReturn= []
    for l in level2[:6]:
        M = cv.moments(l)
        cX = int(M["m10"] / (M["m00"] + 0.00000001))
        cY = int(M["m01"] / (M["m00"] + 0.00000001))
        toReturn.append((cX+x,cY+y))
    if debug:
        for r in toReturn:
            img[r[1]][r[0]] = [0,255,0]
            img[r[1]+1][r[0]+1] = [0,255,0]
            img[r[1]-1][r[0]-1] = [0,255,0]
            img[r[1]+1][r[0]-1] = [0,255,0]
            img[r[1]-1][r[0]+1] = [0,255,0]
        show(img)
    #toReturn = [ l[0][0] for l in level2[:6]]
    #toReturn = [ (x1+x,y1+y) for (x1,y1) in toReturn]
    #print(level2[0][0][0])
    #print('edges shape')
    #print(len( level2))
    #print(len( level2[0]))
    #print(len( level2[0][0]))
    print(toReturn)
    return toReturn[0]
    #cv.imshow('output',orgRect)

