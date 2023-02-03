from imutils.perspective import four_point_transform 
from imutils import contours 
import imutils 
import cv2 
from numpy import reshape


def scale_up(img):
    scale_percent = 90 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

#(number in 7 segment dispaly): Actual Digit 
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

#Note: imread loads only ONE image at a time right now
#Note 2: Could put this in a loop but need to figure out how to load multiple images just taken 
def get_digits(image): 
    image = cv2.imread("1.7.png")
    image = scale_up(image)
    
    #image = imutils.resize(image, height = 500) #Resizes the Image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50 ,200, 255)
    #cv2.imshow("Image1", edged)
    #cv2.waitKey(0)


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
    '''

    img2 = cv2.imread("opentest.png")
    output1 = four_point_transform(img2,dispaly.reshape(4,2))
    cv2.imshow("Image2", output1)
    cv2.waitKey(0)
    '''

    thresh = cv2.threshold(warped, 0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
    #thresh = cv2.threshold(warped, 0,255,cv2.THRESH_BINARY)[1]


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    cv2.imshow("Image", thresh)
    cv2.waitKey(0)
#---------------------------------------------------------------------------------------------------------------------#


    items = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(items)

    digitcnts = [] #empty list? 
    for c in cnts: 
        (x,y,w,h) = cv2.boundingRect(c) #each letter might corresbond to some part of the bounding rectangle 
        rect = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow("Output", rect)
        cv2.waitKey(0)
        if w >= 15 and (h>= 30 and h<=40): 
            digitcnts.append(c) # append the contour if it is big enough 
    #---------------(Note)-----------------# 
    # h = height, w = width of the bounding box. These parameters will probably need to be tuned 
    digitcnts = contours.sort_contours(digitcnts, method="left-to-right")[0]
    digits = []
    for c in digitcnts:
	# extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]
        cv2.imshow("Num", roi)
        cv2.waitKey(0)
        # cv2.imshow("k", roi)
        # cv2.waitKey(0)
        # compute the width and height of each of the 7 segments
        # we are going to examine
        (roiH, roiW) = roi.shape
        #print(h,w)
        #print(roiH,roiW)
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.05)

        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),	# top
            ((0, 0), (dW, h // 2)),	# top-left
            ((w - dW, 0), (w, h // 2)),	# top-right
            ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
            ((0, h // 2), (dW, h)),	# bottom-left
            ((w - dW, h // 2), (w, h)),	# bottom-right
            ((0, h - dH), (w, h))	# bottom
        ]
        on = [0] * len(segments)

        # contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for contour in contours:
        #     cv2.drawContours(output, [contour], 0, (0,255,0), 3)

        # for x in segments:
        #     print(x)


        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            #print(segROI)
            # if the total number of non-zero pixels is greater than
            # 50% of the area, mark the segment as "on"
            if total / float(area) > 0.5:
                on[i]= 1
                
        # lookup the digit and draw it on the image
        digit = DIGITS_LOOKUP[tuple(on)]
        digits.append(digit)
        print(digit)


    
        









    
    #cv2.drawContours(digitcnts[0], cnts, -1, (0, 255, 0), 3)
    #cv2.imshow('Contours', digitcnts[0])
    #cv2.waitKey(0)
    
    


   














def main():
    get_digits()
    

if __name__ == "__main__":
    main()

