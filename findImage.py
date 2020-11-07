import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import pyautogui
import imutils
def imageToText():
    img = cv.imread('./o21.png',0)
    ret,source = cv.threshold(img,127,255,cv2.THRESH_BINARY)
    source = (255 - source) 
    cv.imshow('image',source)
    cv.waitKey(0)
    cv.destroyAllWindows()
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_rgb = Image.frombytes('RGB', img.shape[:2], img, 'raw', 'BGR', 0, 0)
    #print(pytesseract.image_to_string(img,config='digits'))

#time.sleep(2)
#screen = pyautogui.screenshot()
#img = cv.cvtColor(np.array(screen), cv.COLOR_RGB2GRAY)
#img = cv.imread('./allnumbers.png',0)
img = cv.imread('./o21.png',0)
color = cv.imread('./o21.png')
#img = cv.imread('./2o.png',0)


#img = cv.resize(img,(15,10),interpolation = cv.INTER_AREA)
#
#rows = img.shape[0]
#cols = img.shape[1]
#img_center = (cols / 2, rows / 2)
#M = cv.getRotationMatrix2D(img_center, 25, 1)
#rotated_image = cv.warpAffine(img, M, (cols, rows),borderValue=(255,255,255))
#
ret,source = cv.threshold(img,127,255,cv2.THRESH_BINARY)
cv.imshow('image',source)
cv.waitKey(0)
cv.destroyAllWindows()
#exit()

#screen = pyautogui.screenshot()
#img = cv.cvtColor(np.array( screen ), cv.COLOR_RGB2GRAY)

#img = cv.imread('./allnumbers.png',0)
img2 = source.copy()
#template = cv.imread('./8n.png',0)
#w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
methods = methods[1:2]
for meth in methods:
    for i in range(0,10):
        template = cv.imread('./' +str(i) + 'n.png',0)
        print(i)
        ret,temp = cv2.threshold(template,127,255,cv2.THRESH_BINARY)
        #temp = cv.resize(temp,(15,10),interpolation = cv.INTER_AREA)
        #rows = temp.shape[0]
        #cols = temp.shape[1]
        #img_center = (cols / 2, rows / 2)
        #M = cv.getRotationMatrix2D(img_center, 25, 1)
        #temp = cv.warpAffine(temp, M, (cols, rows),borderValue=(255,255,255))
        w, h = temp.shape[::-1]
        img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img,temp,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        #if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            #top_left = min_loc
        #else:
        imgColor = color.copy()
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(imgColor,top_left, bottom_right, 255, 4)
        print(top_left)
        print(bottom_right)
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(imgColor)
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()

