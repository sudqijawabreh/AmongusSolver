import pyautogui
import keyboard
import copy
import time
def Use():
    use = pyautogui.Point(x=1273, y=658)
    pyautogui.click(use.x, use.y)
    time.sleep(1)

def adminSwipe():
    homecard = pyautogui.Point(x=530, y=583)
    second = pyautogui.Point(x=347, y=296)
    final = pyautogui.Point(x=1148, y=285)
    time.sleep(2)
    Use()
    pyautogui.click(homecard.x, homecard.y)
    time.sleep(1)
    pyautogui.moveTo(second.x, second.y) 
    pyautogui.dragTo(final.x, final.y, duration = 0.7) 


def GetWireColors(x1,x2,y,diff):
    wires = []
    for i in range(0, 4):
        leftColor =(pyautogui.pixel(x1, y + diff * i))
        rightColor =(pyautogui.pixel(x2, y + diff * i))
        wires.append(((x1, y + diff * i), leftColor))
        wires.append(((x2, y + diff * i), rightColor))
    return wires

def ConnectWires(wires):
   for i in range(0,len(wires),2):
        ((lx, ly), c1 )= wires[i]
        ((rx, ry), c2 )= wires[i + 1]
        print((lx,ly),c1)
        print((rx,ry),c2)
        pyautogui.moveTo(lx, ly)
        pyautogui.dragTo(rx, ry, duration = 0.45)

def FixWires():
    diff = 132
    x1 = 385
    x2 = 960
    y1 = 190
    time.sleep(2)
    Use()
    wires = GetWireColors(x1,x2,y1,diff)
    sortedWires = sorted(wires, key = lambda w: (w[1],w[0][0]))
    #print(sortedWires)
    ConnectWires(sortedWires)

def Trash():
    exitTrash = pyautogui.Point(x=330, y=91)
    time.sleep(2)
    Use()
    init = pyautogui.Point(x=900, y=297)
    final = pyautogui.Point(x=911, y=515)
    pyautogui.moveTo(init.x, init.y)
    pyautogui.mouseDown()
    pyautogui.moveTo(final.x, final.y, duration=0.1)
    time.sleep(1.4)
    pyautogui.mouseUp()
    #pyautogui.click(exitTrash.x, exitTrash.y)



def Download():
    time.sleep(2)
    Use()
    init = pyautogui.Point(x=677, y=467)
    pyautogui.click(init.x, init.y)

def reversKeys(keys):
    ckeys = copy.copy(keys)
    arrows = ['left','right','up','down']
    revers = {'left':'right', 'right':'left', 'up':'down', 'down':'up'}
    for key in ckeys:
        if key.name in arrows:
            key.name = revers[key.name]
            key.event_type = revers[key.event_type]
    return ckeys


def GetBoxes(n):
    uniqueBox = set()
    boxes = []
    while 1:
        box = pyautogui.locateOnScreen('./blue.png', grayscale=True , region=(339,296,300,400),confidence = 0.9)
        #print(box)
        print(len(boxes))
        if box is not None and box:
            uniqueBox.add(box)
            boxes.append(box)
            time.sleep(0.3)
        if len(boxes) == n:
            break;
    return boxes

def ClickBoxes(boxes):
    mirrorx = 442
    for box in boxes:
        pyautogui.click(box.left + 10 + mirrorx, box.top + 10)
        time.sleep(0.2)

def Reactor():
    for i in range(1,6):
        boxes = GetBoxes(i)
        time.sleep(0.5)
        ClickBoxes(boxes)

def Cycles():
    time.sleep(2)
    circelToButtonDiffY = 218 - 183
    circelToButtonDiffX = 876 - 574
    diffY = 375 - 183
    init = pyautogui.Point(574, 183)
    count = 0
    while 1:
        r,g,b = pyautogui.pixel(init.x,init.y)
        if r in range(100) and b in range(100) and b in range(100):
            pyautogui.click(init.x + circelToButtonDiffX, init.y + circelToButtonDiffY)
            print('bingo')
            init = pyautogui.Point(init.x, init.y + diffY)
            count +=1
        if count == 3:
            break


def StartNumbers():
    for i in range(1,10):
        number = pyautogui.locateOnScreen(str(i) + '.png', grayscale= True ,confidence=0.9)
        if not (number is None):
            pyautogui.click(number.left+20, number.top+20)

def AIStartNumber():
    import cv2 as cv
    import numpy as np
    from matplotlib import pyplot as plt
    time.sleep(1)
    Use()
    screen = pyautogui.screenshot()
    img = cv.cvtColor(np.array(screen), cv.COLOR_RGB2GRAY)
    ret,source = cv.threshold(img,127,255,cv.THRESH_BINARY)
    #img = cv.imread('./allnumbers.png',0)
    img2 = source.copy()
    #template = cv.imread('./8n.png',0)
    #w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    meth = 'cv.TM_CCOEFF'
    numberLocations = []
    for i in range(1,11):
        template = cv.imread('./' +str(i) + 'n.png',0)
        ret,temp = cv.threshold(template,127,255,cv.THRESH_BINARY)
        w, h = temp.shape[::-1]
        img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img,temp,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        #bottom_right = (top_left[0] + w, top_left[1] + h)
        #cv.rectangle(img,top_left, bottom_right, 255, 2)
        #print(top_left)
        #print(bottom_right)
        #plt.subplot(121),plt.imshow(res,cmap = 'gray')
        #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        #plt.subplot(122),plt.imshow(img,cmap = 'gray')
        #plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        #plt.suptitle(meth)
        #plt.show()
        numberLocations.append(top_left)
    for location in numberLocations:
        pyautogui.click(location[0], location[1])

def SolveO2():
    import contrs
    import cv2 as cv
    import numpy as np
    time.sleep(2)
    #Use()
    screen = pyautogui.screenshot()
    img = cv.cvtColor(np.array(screen), cv.COLOR_RGB2GRAY)
    print(contrs.GetO2NumKeys(img))
    print(img.shape)
    print(contrs.GetO2Numbers(img))

print("hello")
