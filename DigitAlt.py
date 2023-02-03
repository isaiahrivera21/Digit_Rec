from imutils.perspective import four_point_transform 
from imutils import contours 
import imutils 
import cv2 
from numpy import reshape
import numpy as np
from Seg7 import Segments
from PIL import Image, ImageChops 

def scale_up(img):
    scale_percent = 250 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized



#Note: imread loads only ONE image at a time right now
#Note 2: Could put this in a loop but need to figure out how to load multiple images just taken 
def get_digits(): 
    image = cv2.imread("9.0.png")
    #image = imutils.resize(image, height = 500) #Resizes the Image
    image = scale_up(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50 ,200, 255)

    

    items = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(items)
    cnts = sorted(cnts, key = cv2.contourArea, reverse=True)
    dispaly = None 


    for c in cnts: 
            arc_Length = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c, 0.02 * arc_Length, True)
            if len(approx) == 4: 
                dispaly = approx
                break 

    #print(type(dispaly))
    
    warped = four_point_transform(gray,dispaly.reshape(4,2))
    output = four_point_transform(image,dispaly.reshape(4,2))
    cv2.imshow("Output", output)
    cv2.waitKey(0)

    '''
        lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)

        cv2.imshow("lab", lab)
        cv2.waitKey(0)

        kernel = np.ones((5,5), np.uint8)

        # threshold params
        low = 165
        high = 200
        iters = 3

        # make copy
        copy = b.copy()

        thresh = cv2.inRange(copy, low, high)

        # dilate
        for a in range(iters):
            thresh = cv2.dilate(thresh, kernel)

        # erode
        for a in range(iters):
            thresh = cv2.erode(thresh, kernel)

        # show image
        cv2.imshow("thresh", thresh)
        #cv2.imwrite("threshold.jpg", thresh)
        
    '''

    '''

    img2 = cv2.imread("opentest.png")
    output1 = four_point_transform(img2,dispaly.reshape(4,2))
    cv2.imshow("Image2", output1)
    cv2.waitKey(0)
    '''

    thresh = cv2.threshold(warped, 0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #thresh = ~thresh

    iters = 3


    # cv2.imshow("thresh2", thresh)
    # cv2.waitKey(0)

    for a in range(iters):
        thresh = cv2.dilate(thresh, kernel)
        # cv2.imshow("k1", thresh)
        # cv2.waitKey(0)

    # erode
    for a in range(iters):
        thresh = cv2.erode(thresh, kernel)
        # cv2.imshow("k2", thresh)
        # cv2.waitKey(0)


    '''cv2.imwrite('savedImage.jpg', thresh)

    img = Image.open('savedImage.jpg')
    inv_img = ImageChops.invert(img)
    inv_img.show()'''

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(output, [contour], 0, (0,255,0), 3)

    bounds = []
    h, w = image.shape[:2]
    for contour in contours:
        left = w
        right = 0
        top = h
        bottom = 0
        for point in contour:
            point = point[0]
            x, y = point
            if x < left:
                left = x
            if x > right:
                right = x
            if y < top:
                top = y
            if y > bottom:
                bottom = y
        tl = [left, top]
        br = [right, bottom]
        bounds.append([tl, br])

    # crop out each number
    cuts = []
    number = 0
    for bound in bounds:
        tl, br = bound
        cut_img = thresh[tl[1]:br[1], tl[0]:br[0]]
        cuts.append(cut_img)
        number += 1
        #cv2.imshow(str(number), cut_img)
        #cv2.waitKey(0)
#---------------------------------------------------------------------------------------------------------------------#
    model = Segments()
    index = 0
    
    for x in range(len(cuts)):
        # save image
        #cv2.imwrite(str(index) + "_" + str(number) + ".jpg", cut)

        # process
        model.digest(cuts[x])
        number = model.getNum() #THIS IS THING WE PASS TO OTHER PROGRAM 

        #showing values guessed by the system 
        print(number)
        cv2.imshow(str(index), cuts[x])
        cv2.waitKey(0)
    cv2.imshow("contours", output)
    cv2.waitKey(0)

        
    

def main():
    get_digits()

if __name__ == "__main__":
    main()

#0: -1
#1: Works
#2: Works
#3: -1  Works now 
#4: -1 
#5: Works (Thinks its a one)
#6: -1 
#7: Not tested 
#8: Works
#9: -1 