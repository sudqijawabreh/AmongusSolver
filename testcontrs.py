import cv2 as cv
import contrs



def testO2Nums(debug):
    contrs.debug = debug
    files = ['o21.png',
            'o22.png',
            'o23.png',
            'o24.png',
            'red.png',
            'nored.png',
            'wrong1.png',
            'wrong4.png']
    expectedValues =[[8,1,9,2,8],
                     [2,5,8,5,3],
                     [6,4,4,1,3],
                     [0,4,0,2,5],
                     [5,4,7,9,3],
                     [5,4,7,9,3],
                     [9,1,4,4,4],
                     [7,1,9,1,1]]
    #print(len(files))
    #print(len(expectedValues))

    for (f,expected) in zip(files,expectedValues):
        img = cv.imread('./images/'+f)
        actual = contrs.GetO2Numbers(img)
        if not (actual == expected):
            print('fail ' + f)
            print('expected ' + str(expected))
            print('got ' + str(actual))
        else:
            print(actual)

testO2Nums(True)
